# =============================================================================
# Highlight all System Blocks in Innovus 15.2 matching the Architecture Diagram
# Usage in Innovus: source highlight_blocks.tcl
# (Tested on out/05_final.enc.dat; uses selectInst + highlight -index)
# =============================================================================
dehighlight
deselectAll

# 1. TCDM Banks 0..15 (Weight Ring / Pinned KV-Cache - 144 KB)
for {set i 0} {$i < 16} {incr i} {
    selectInst u_tcdm/GEN_BANKS_${i}__u_bank/*
}
highlight -index 1
deselectAll

# 2. TCDM Banks 16..31 (Ping-Pong Acc/O & Output Buffers - 144 KB)
for {set i 16} {$i < 32} {incr i} {
    selectInst u_tcdm/GEN_BANKS_${i}__u_bank/*
}
highlight -index 2
deselectAll

# 3. 25x32 Logarithmic Interleaved Interconnect
selectInst u_interconnect/*
highlight -index 3
deselectAll

# 4. Data Streamers (Weight, Act/Q, FP16 Read A/B, 4D Write-Out)
selectInst u_wt_streamer/*
selectInst u_act_streamer/*
selectInst u_read_fp16/*
selectInst u_read_fp16_b/*
selectInst u_write_out/*
highlight -index 4
deselectAll

# 5. Unified Sequencer + CSR Config
selectInst u_seq/*
selectInst u_csr/*
highlight -index 5
deselectAll

# 6. Log8-FP4 Vector MAC Array + 1W2R D-Latch + Deferred O Rescaler
selectInst u_core/u_mac_array/*
selectInst u_core/u_acc_buffer/*
selectInst u_core/u_attn_rescaler/*
highlight -index 6
deselectAll

# 7. On-Chip FlashAttention Loop (Exp-Free Softmax + P-FIFO + Denom. Accum)
selectInst u_core/u_softmax_requantizer/*
selectInst u_core/u_p_fifo/*
selectInst u_core/u_l_accumulator/*
highlight -index 7
deselectAll

# 8. Lazy FP16 Drain Normalizer
selectInst u_core/u_final_normalizer/*
highlight -index 8
deselectAll

# 9. Custom-FP20 SFU (+ SFU Sequencer)
selectInst u_sfu/*
selectInst u_sfu_seq/*
highlight -index 9
deselectAll

# 10. Log-Quantizer (SLQ Encoder)
selectInst u_log_quantizer/*
highlight -index 10
deselectAll

# 11. 16x16 V-Transpose Buffer (FP16)
selectInst u_transpose_buffer/*
selectInst u_transpose_ctrl/*
highlight -index 11
deselectAll

puts "INFO: Highlighted all 11 system block groups!"
