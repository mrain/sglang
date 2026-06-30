//! sglang-server: a multi-threaded Rust frontend (API server → TokenizerManager
//! → Tokenizer/Detokenizer) embedded in the Python scheduler process.
//!
//! Pipeline stages 1–5 are pure Rust and never touch a `PyObject`, so they run
//! concurrently with the Python scheduler without contending for the GIL. The
//! only GIL crossings are the two boundary methods on [`Server`]:
//!   * `recv_requests` — Python scheduler thread drains the ingress ring.
//!   * `push_chunk`    — Python scheduler thread pushes one output chunk.
//!
//! Both are non-blocking, so the GIL is never held across a wait.

mod api_server;
pub mod config;
mod detokenizer;
mod error;
mod fsm;
mod ids;
mod message;
pub mod observability;
mod runtime;
pub mod schema;
pub mod state;
mod tokenizer;
mod tokenizer_manager;

use std::net::SocketAddr;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::cmp::max;

use std::sync::Arc;

use crate::runtime::{Runtime, RuntimeConfig};

/// Columnar ingress batch handed to Python by [`Server::recv_requests`]:
/// `(headers, ids_buf, lengths)` — per-request scalar msgpack headers, all
/// requests' raw int64 ids concatenated, and per-request token counts.
type IngressBatch<'py> = (Vec<Bound<'py, PyBytes>>, Bound<'py, PyBytes>, Vec<u32>);

/// Handle owned by the Python scheduler process. Construct once via
/// [`Server::start`], then poll it from the scheduler event loop.
#[pyclass]
struct Server {
    rt: Runtime,
}

#[pymethods]
impl Server {
    /// Boot the frontend (spawns all threads) and return immediately.
    #[new]
    #[pyo3(signature = (
        bind = None,
        api_worker_num = None,
        tokenizer_worker_num = None,
        detokenizer_worker_num = None,
        ingress_ring_cap = 8192,
        egress_ring_cap = 8192,
        channel_cap = 8192,
        pin_cores = true,
        cores = None,
        tokenizer_path = None,
        revision = None,
        server_args_json = "{}",
    ))]
    // pyo3 `#[new]` constructor: the wide arg list is the Python-facing boot
    // surface (all optional overrides), not a call-site ergonomics problem.
    #[allow(clippy::too_many_arguments)]
    fn start(
        bind: Option<String>,
        api_worker_num: Option<usize>,
        tokenizer_worker_num: Option<usize>,
        detokenizer_worker_num: Option<usize>,
        ingress_ring_cap: usize,
        egress_ring_cap: usize,
        channel_cap: usize,
        pin_cores: bool,
        cores: Option<Vec<usize>>,
        tokenizer_path: Option<String>,
        revision: Option<String>,
        server_args_json: &str,
    ) -> PyResult<Self> {
        // Static server metadata (server_args + model_config) dumped by the
        // scheduler; parse and validate mandatory fields now so a bad/missing
        // field is a boot error, not a request-time 500.
        let server_args: runtime::ServerArgs = runtime::ServerArgs::from_json(server_args_json)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "bad server_args_json: {e}"
                ))
            })?;
        server_args.validate_mandatory().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("server_args: {e}"))
        })?;
        // The bind address, tokenizer source/threads/shards all live in the
        // `server_args` blob; resolve them from there so the scheduler doesn't
        // re-pass them. The explicit params stay as optional overrides for
        // standalone callers (tests) that construct a `Server` without a full
        // `server_args`.
        let bind: SocketAddr = bind
            .or_else(|| server_args.bind())
            .unwrap_or_else(|| "127.0.0.1:30000".to_string())
            .parse()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad bind: {e}"))
            })?;

        let tokenizer_worker_num =
            tokenizer_worker_num.unwrap_or_else(|| server_args.tokenizer_worker_num());
        let detokenizer_worker_num =
            detokenizer_worker_num.unwrap_or_else(|| server_args.detokenizer_worker_num());
        let api_worker_num = api_worker_num
            .unwrap_or_else(|| max(4, max(tokenizer_worker_num / 2, detokenizer_worker_num / 2)));

        let tokenizer_path = tokenizer_path.or_else(|| server_args.tokenizer_path());
        let revision = revision.or_else(|| server_args.revision());
        let server_args = std::sync::Arc::new(server_args);

        let cfg = RuntimeConfig {
            bind,
            api_worker_num,
            tokenizer_worker_num,
            detokenizer_worker_num,
            ingress_ring_cap,
            egress_ring_cap,
            channel_cap,
            pin_cores,
            cores,
            tokenizer_path,
            revision,
            startup_tx: None,
            server_args,
        };
        let rt = runtime::start(cfg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("runtime start failed: {e}"))
        })?;
        Ok(Server { rt })
    }

    /// Non-blocking drain of the ingress ring, returned **columnar** so the large
    /// `input_ids` tensor never goes through msgpack. Yields a 3-tuple:
    ///   * `headers`: `list[bytes]` — one msgpack scalar header per request
    ///     (`input_ids` omitted), decoded individually by the scheduler;
    ///   * `ids_buf`: `bytes` — all requests' raw little-endian int64 ids,
    ///     concatenated; sliced per request and wrapped as `array("q")`;
    ///   * `lengths`: `list[int]` — per-request token count (0 for control reqs),
    ///     so the scheduler can slice `ids_buf`.
    /// The GIL is released for the drain + columnar split; only the `PyBytes`
    /// marshaling needs it. The `ids` cells are copied **directly into the result
    /// `bytes`** (one copy, no intermediate buffer).
    #[pyo3(signature = (max = 256))]
    fn recv_requests<'py>(&self, py: Python<'py>, max: usize) -> PyResult<IngressBatch<'py>> {
        let cols = py.detach(|| self.rt.ingress.drain(max));
        let headers = cols.headers.iter().map(|h| PyBytes::new(py, h)).collect();
        // Single pass: copy each raw ids cell straight into the output `bytes`.
        let ids_buf = PyBytes::new_with(py, cols.ids_total, |buf| {
            let mut pos = 0;
            for cell in &cols.ids {
                let end = pos + cell.len();
                buf[pos..end].copy_from_slice(cell);
                pos = end;
            }
            Ok(())
        })?;
        Ok((headers, ids_buf, cols.lengths))
    }

    /// Push one scheduler-output chunk (already msgpack-encoded `ChunkEvent`)
    /// into the egress ring → detok shard. Returns `False` on backpressure.
    fn push_chunk(&self, py: Python<'_>, chunk: &[u8]) -> bool {
        let bytes = crate::message::frame_egress_chunk(chunk);
        py.detach(|| self.rt.egress.try_push(bytes))
    }

    /// Push a control-request result (e.g. the `/server_info` JSON) into the
    /// egress ring, routed by `rid` to the waiting request's sink as a single
    /// non-streamed response. Returns `False` on backpressure.
    fn push_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        let bytes = crate::message::frame_egress_result(rid, payload);
        py.detach(|| self.rt.egress.try_push(bytes))
    }

    /// Signal all threads to stop (best effort).
    fn shutdown(&self) {
        self.rt.request_shutdown();
    }
}

// ── TokenizerManager PyO3 class ──

/// Rust TokenizerManager — replaces the Python `TokenizerManager`.
/// Constructed from the same config JSON blob the Python scheduler dumps.
#[pyclass]
struct TokenizerManager {
    rt: Runtime,
}

#[pymethods]
impl TokenizerManager {
    #[new]
    #[pyo3(signature = (server_args_json, startup_timeout_ms = 5000))]
    fn new(server_args_json: &str, startup_timeout_ms: u64) -> PyResult<Self> {
        let cfg = config::RuntimeConfigView::from_json(server_args_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad config: {e}"))
        })?;

        let server_args = Arc::new(runtime::ServerArgs::from_json(server_args_json).map_err(
            |e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad server_args: {e}")),
        )?);

        let bind = cfg.bind_addr().parse().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad bind: {e}"))
        })?;

        // Startup handshake: API server signals bind success/failure
        let (startup_tx, startup_rx) = std::sync::mpsc::channel();

        let runtime_cfg = runtime::RuntimeConfig {
            bind,
            api_worker_num: max(
                4,
                max(cfg.tokenizer_worker_num / 2, cfg.detokenizer_worker_num / 2),
            ),
            tokenizer_worker_num: cfg.tokenizer_worker_num,
            detokenizer_worker_num: cfg.detokenizer_worker_num,
            ingress_ring_cap: 8192,
            egress_ring_cap: 8192,
            channel_cap: 8192,
            pin_cores: true,
            cores: None,
            tokenizer_path: cfg.resolved_tokenizer_path(),
            revision: if cfg.revision.is_empty() {
                None
            } else {
                Some(cfg.revision.clone())
            },
            startup_tx: Some(startup_tx),
            server_args,
        };

        let rt = runtime::start(runtime_cfg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("runtime start failed: {e}"))
        })?;

        // Wait for API server to bind (or fail) within the timeout
        match startup_rx.recv_timeout(std::time::Duration::from_millis(startup_timeout_ms)) {
            Ok(Ok(())) => {} // bind succeeded, proceed
            Ok(Err(msg)) => {
                rt.request_shutdown();
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                rt.request_shutdown();
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "API server did not start within {}ms",
                    startup_timeout_ms
                )));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // Channel disconnected without a message — runtime thread panicked
                rt.request_shutdown();
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "API server thread terminated unexpectedly".to_string(),
                ));
            }
        }

        Ok(TokenizerManager { rt })
    }

    /// Non-blocking drain of the ingress ring.
    #[pyo3(signature = (max = 256))]
    fn recv_requests<'py>(&self, py: Python<'py>, max: usize) -> PyResult<IngressBatch<'py>> {
        let cols = py.detach(|| self.rt.ingress.drain(max));
        let headers = cols.headers.iter().map(|h| PyBytes::new(py, h)).collect();
        let ids_buf = PyBytes::new_with(py, cols.ids_total, |buf| {
            let mut pos = 0;
            for cell in &cols.ids {
                let end = pos + cell.len();
                buf[pos..end].copy_from_slice(cell);
                pos = end;
            }
            Ok(())
        })?;
        Ok((headers, ids_buf, cols.lengths))
    }

    /// Push one generation chunk into the egress ring.
    fn push_chunk(&self, py: Python<'_>, chunk: &[u8]) -> bool {
        let bytes = crate::message::frame_egress_chunk(chunk);
        py.detach(|| self.rt.egress.try_push(bytes))
    }

    /// Push a control result into the egress ring.
    fn push_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        let bytes = crate::message::frame_egress_result(rid, payload);
        py.detach(|| self.rt.egress.try_push(bytes))
    }

    /// Send an abort request through the ingress ring to the scheduler.
    /// The scheduler will remove the request and echo back an echo so the
    /// TM can clean up local state.
    fn abort_request(&self, rid: &str) -> PyResult<bool> {
        use crate::message::control_req_msgpack;
        let header = control_req_msgpack("AbortReq", rid)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(self.rt.push_control_to_ingress(header))
    }

    /// Signal all threads to stop.
    fn shutdown(&self) {
        self.rt.request_shutdown();
    }
}

/// The Python module: `import sglang_server`.
#[pymodule]
fn sglang_server(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    m.add_class::<Server>()?;
    m.add_class::<TokenizerManager>()?;
    Ok(())
}
