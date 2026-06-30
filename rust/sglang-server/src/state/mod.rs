//! Request lifecycle state: per-request accumulator, pending response table,
//! and type-based response dispatcher.

pub mod dispatcher;
pub mod pending;
pub mod req_state;
pub mod server_status;
