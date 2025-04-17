var<workgroup> lock: atomic<u32>;
@group(0) @binding(0)
var<storage, read_write> ctr: u32;

@compute @workgroup_size(2, 1, 1)
fn loop_continue(@builtin(global_invocation_id) id: vec3u) {
    loop {
        if atomicOr(&lock, 1u) == 1u { continue; }
        ctr++;
        atomicAnd(&lock, 0u);
        break;
    }
}

@compute @workgroup_size(2, 1, 1)
fn loop_break(@builtin(global_invocation_id) id: vec3u) {
    loop {
        if atomicOr(&lock, 1u) == 0u { break; }
    }
    ctr++;
    atomicAnd(&lock, 0u);
}

@compute @workgroup_size(2, 1, 1)
fn while_other_has_lock(@builtin(global_invocation_id) id: vec3u) {
    while atomicOr(&lock, 1u) == 1u { }
    ctr++;
    atomicAnd(&lock, 0u);
}

@compute @workgroup_size(2, 1, 1)
fn while_self_has_not_lock(@builtin(global_invocation_id) id: vec3u) {
    while atomicOr(&lock, 1u) != 0u { }
    ctr++;
    atomicAnd(&lock, 0u);
}
