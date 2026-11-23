# Full-layer (prefill) system TB for snax_llm_top_clean
# Paths are relative to sim/
+incdir+../01_snax_llm_fp4_subsystem

../01_snax_llm_fp4_subsystem/fp4_mult_lut.sv
../01_snax_llm_fp4_subsystem/compressor_4to2.sv
../01_snax_llm_fp4_subsystem/dot_product_16.sv
../01_snax_llm_fp4_subsystem/latch_array.sv
../01_snax_llm_fp4_subsystem/vsq_rescaler_32b.sv
../01_snax_llm_fp4_subsystem/log8_fp4_arith_units.sv
../01_snax_llm_fp4_subsystem/vsq_matmul_engine.sv
../01_snax_llm_fp4_subsystem/softmax_requantizer.sv
../01_snax_llm_fp4_subsystem/p_fifo.sv
../01_snax_llm_fp4_subsystem/l_accumulator.sv
../01_snax_llm_fp4_subsystem/final_normalizer.sv
../01_snax_llm_fp4_subsystem/slq_encoder.sv
../01_snax_llm_fp4_subsystem/log8_fp4_mac_array.sv
../01_snax_llm_fp4_subsystem/rmw_latch_acc_buffer.sv
../01_snax_llm_fp4_subsystem/attn_o_rescaler_bank.sv
../01_snax_llm_fp4_subsystem/snax_sfu_fp20_units.sv
../01_snax_llm_fp4_subsystem/snax_llm_sfu_opt_pipe.sv
../01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv
../01_snax_llm_fp4_subsystem/skid_buffer.sv
../01_snax_llm_fp4_subsystem/snax_streamer_act_bcast.sv
../01_snax_llm_fp4_subsystem/snax_streamer_wt_16ch.sv
../01_snax_llm_fp4_subsystem/snax_streamer_write_out.sv
../01_snax_llm_fp4_subsystem/snax_transpose_buffer.sv
../01_snax_llm_fp4_subsystem/snax_transpose_ctrl.sv
../01_snax_llm_fp4_subsystem/snax_stream_read_fp16.sv
../01_snax_llm_fp4_subsystem/snax_csr_manager.sv
../01_snax_llm_fp4_subsystem/snax_flash_csr_snax.sv
../01_snax_llm_fp4_subsystem/snax_seq_dr_fsm.sv
../01_snax_llm_fp4_subsystem/snax_seq_dma_fsm.sv
../01_snax_llm_fp4_subsystem/snax_seq_cmp_fsm.sv
../01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv
../01_snax_llm_fp4_subsystem/fixed_priority_arb_n.sv
../01_snax_llm_fp4_subsystem/interleaved_interconnect_72b.sv
../01_snax_llm_fp4_subsystem/multi_bank_tcdm_72b.sv
../01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv

../tb/tb_snax_llm_layer_smoke.sv
