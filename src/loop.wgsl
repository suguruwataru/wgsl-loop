var<workgroup> atomic_compare_exchange_weak_var: atomic<u32>;
@compute @workgroup_size(1, 1, 1)
fn atomic_compare_exchange_weak() {
    atomicCompareExchangeWeak(&atomic_compare_exchange_weak_var, 0u, 1u);
}
