namespace xf {
namespace cv {

template <int SRC_T,
          int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int USE_URAM = 0,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int WORDWIDTH,
          int SRC_TC,
          int PLANES>
void xFHistogramKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
                       uint32_t hist_array[PLANES][256],
                       uint16_t& imgheight,
                       uint16_t& imgwidth) {
    /* L1: */
#pragma HLS array_partition variable=hist_array type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}

    // Temporary array used while computing histogram
L2: uint32_t tmp_hist[(PLANES << XF_BITSHIFT(NPC))][256];
#pragma HLS array_partition variable=tmp_hist type=auto{_ARRAY_T_L2} factor=auto{_ARRAY_F_L2} dim=auto{_ARRAY_D_L2}

L3: uint32_t tmp_hist1[(PLANES << XF_BITSHIFT(NPC))][256];
#pragma HLS array_partition variable=tmp_hist1 type=auto{_ARRAY_T_L3} factor=auto{_ARRAY_F_L3} dim=auto{_ARRAY_D_L3}

    XF_SNAME(WORDWIDTH) in_buf, in_buf1, temp_buf;

    bool flag = 0;

L4: HIST_INITIALIZE_LOOP:
    for (ap_uint<10> i = 0; i < 256; i++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
L5:     for (ap_uint<5> j = 0; j < ((1 << XF_BITSHIFT(NPC)) * PLANES); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=256 max=256
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
            // clang-format on
            tmp_hist[j][i] = 0;
            tmp_hist1[j][i] = 0;
        }
    }

L6: HISTOGRAM_ROW_LOOP:
    for (ap_uint<13> row = 0; row < imgheight; row++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}
    // clang-format on
L7: HISTOGRAM_COL_LOOP:
        for (ap_uint<13> col = 0; col < (imgwidth); col = col + 2) {
// clang-format off
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS LOOP_TRIPCOUNT min=SRC_TC max=SRC_TC
#pragma HLS pipeline II=auto{_PIPE_L7}
#pragma HLS unroll factor=auto{_UNROLL_L7}
            // clang-format on
            in_buf = _src_mat.read(row * (imgwidth) + col);

            if (col == (imgwidth - 1))
                in_buf1 = 0;
            else
                in_buf1 = _src_mat.read(row * (imgwidth) + col + 1);

// clang-format off
#pragma HLS DEPENDENCE variable=tmp_hist array intra false
#pragma HLS DEPENDENCE variable=tmp_hist1 array intra false
        // clang-format on
L8: EXTRACT_UPDATE:
            for (ap_uint<9> i = 0, j = 0; i < ((8 << XF_BITSHIFT(NPC)) * PLANES); j++, i += 8) {
#pragma HLS pipeline II=auto{_PIPE_L8}
#pragma HLS unroll factor=auto{_UNROLL_L8}
                ap_uint<8> val = 0, val1 = 0;
                val = in_buf.range(i + 7, i);
                val1 = in_buf1.range(i + 7, i);

                uint32_t tmpval = tmp_hist[j][val];
                uint32_t tmpval1 = tmp_hist1[j][val1];
                tmp_hist[j][val] = tmpval + 1;
                if (!(col == (imgwidth - 1))) tmp_hist1[j][val1] = tmpval1 + 1;
            }
        }
    }

    const int num_ch = XF_CHANNELS(SRC_T, NPC);

L9: MERGE_HIST_LOOP:
    for (ap_uint<32> i = 0; i < 256; i++) {
#pragma HLS pipeline II=auto{_PIPE_L9}
#pragma HLS unroll factor=auto{_UNROLL_L9}

L10: MERGE_HIST_CH_UNROLL:
        for (ap_uint<5> ch = 0; ch < num_ch; ch++) {
#pragma HLS pipeline II=auto{_PIPE_L10}
#pragma HLS unroll factor=auto{_UNROLL_L10}
            uint32_t value = 0;

L11: MERGE_HIST_NPPC_UNROLL:
            for (ap_uint<5> p = 0; p < XF_NPIXPERCYCLE(NPC); p++) {
#pragma HLS pipeline II=auto{_PIPE_L11}
#pragma HLS unroll factor=auto{_UNROLL_L11}
                value += tmp_hist[p * num_ch + ch][i] + tmp_hist1[p * num_ch + ch][i];
            }

            hist_array[ch][i] = value;
        }
    }
}

/////////16bit support
template <int SRC_T,
          int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int USE_URAM = 0,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int WORDWIDTH,
          int SRC_TC,
          int PLANES,
          int AEC_HISTSIZE>
void xFHistogramKernel_sin(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                           xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& src2,
                           uint32_t hist[AEC_HISTSIZE],
                           int p,
                           float inputMin,
                           float inputMax,
                           float outputMin,
                           float outputMax) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    /* L12: */
#pragma HLS array_partition variable=hist type=auto{_ARRAY_T_L12} factor=auto{_ARRAY_F_L12} dim=auto{_ARRAY_D_L12}

    const int STEP = XF_DTPIXELDEPTH(SRC_T, NPC);
    int width = src1.cols >> XF_BITSHIFT(NPC);
    int height = src1.rows;
    XF_TNAME(SRC_T, NPC) in_pix, in_pix1, out_pix;
    int writenct = 0;
    int depth = 1;
    int bins = AEC_HISTSIZE;
    int nElements = AEC_HISTSIZE;
    int val;

L13: INITIALIZE_HIST:
    for (int k = 0; k < AEC_HISTSIZE; k++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=AEC_HISTSIZE max=AEC_HISTSIZE
#pragma HLS pipeline II=auto{_PIPE_L13}
#pragma HLS unroll factor=auto{_UNROLL_L13}
        // clang-format on
        hist[k] = 0;
    }

    // Temporary array used while computing histogram
L14: ap_uint<32> tmp_hist[XF_NPIXPERCYCLE(NPC)][AEC_HISTSIZE];
#pragma HLS array_partition variable=tmp_hist type=auto{_ARRAY_T_L14} factor=auto{_ARRAY_F_L14} dim=auto{_ARRAY_D_L14}

    XF_TNAME(SRC_T, NPC) in_buf, in_buf1, temp_buf;
    bool flag = 0;

L15: HIST_INITIALIZE_LOOP:
    for (ap_uint<32> i = 0; i < AEC_HISTSIZE; i++) {
#pragma HLS pipeline II=auto{_PIPE_L15}
#pragma HLS unroll factor=auto{_UNROLL_L15}
L16:     for (ap_uint<5> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=AEC_HISTSIZE max=AEC_HISTSIZE
#pragma HLS pipeline II=auto{_PIPE_L16}
#pragma HLS unroll factor=auto{_UNROLL_L16}
            // clang-format on
            tmp_hist[j][i] = 0;
        }
    }

L17: static uint32_t old[XF_NPIXPERCYCLE(NPC)] = {};
#pragma HLS array_partition variable=old type=auto{_ARRAY_T_L17} factor=auto{_ARRAY_F_L17} dim=auto{_ARRAY_D_L17}

L18: uint32_t acc_rd[XF_NPIXPERCYCLE(NPC)] = {};
#pragma HLS array_partition variable=acc_rd type=auto{_ARRAY_T_L18} factor=auto{_ARRAY_F_L18} dim=auto{_ARRAY_D_L18}

L19: uint32_t acc_wr[XF_NPIXPERCYCLE(NPC)] = {};
#pragma HLS array_partition variable=acc_wr type=auto{_ARRAY_T_L19} factor=auto{_ARRAY_F_L19} dim=auto{_ARRAY_D_L19}

    int readcnt = 0;
    ap_fixed<STEP + 8, STEP + 2> min_vals = inputMin - 0.5f;
    ap_fixed<STEP + 8, STEP + 2> max_vals = inputMax + 0.5f;
    ap_fixed<STEP + 8, STEP + 2> minValue = min_vals, minValue1 = min_vals;
    ap_fixed<STEP + 8, STEP + 2> maxValue = max_vals, maxValue1 = max_vals;
    ap_fixed<STEP + 8, STEP + 2> interval = ap_fixed<STEP + 8, STEP + 2>(maxValue - minValue) / bins;
    ap_fixed<STEP + 8, 2> internal_inv = ((ap_fixed<STEP + 8, 2>)1 / interval);
    int pos = 0, pos1 = 0;
    int currentBin = 0, currentBin1 = 0;

L20: ROW_LOOP:
    for (int row = 0; row != (height); row++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
#pragma HLS pipeline II=auto{_PIPE_L20}
#pragma HLS unroll factor=auto{_UNROLL_L20}
    // clang-format on
L21: COL_LOOP:
        for (int col = 0; col < (width); col = col + 1) {
// clang-format off
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
#pragma HLS pipeline II=auto{_PIPE_L21}
#pragma HLS unroll factor=auto{_UNROLL_L21}

            // clang-format on
            in_pix = src1.read(row * (width) + col);
            src2.write(row * (width) + col, in_pix);

L22:         for (ap_uint<9> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
#pragma HLS DEPENDENCE variable=tmp_hist array intra false
#pragma HLS pipeline II=auto{_PIPE_L22}
#pragma HLS unroll factor=auto{_UNROLL_L22}
                // clang-format on
                XF_CTUNAME(SRC_T, NPC) val = 0, val1 = 0;
                val = in_pix.range(j * STEP + STEP - 1, j * STEP);

                currentBin = int((val - minValue) * internal_inv);

                if (currentBin == old[j]) {
                    acc_rd[j] = acc_wr[j];
                } else {
                    acc_rd[j] = tmp_hist[j][currentBin];
                }

                tmp_hist[j][old[j]] = acc_wr[j];

                acc_wr[j] = acc_rd[j] + 1;
                old[j] = currentBin;
            }
        }
    }

L23: END_HIST_LOOP:
    for (ap_uint<5> ch_ppc = 0; ch_ppc < XF_NPIXPERCYCLE(NPC); ch_ppc++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=NPC
        #pragma HLS pipeline II=auto{_PIPE_L23}
        #pragma HLS unroll factor=auto{_UNROLL_L23}
        // clang-format on
        uint32_t tmp = old[ch_ppc];
        tmp_hist[ch_ppc][tmp] = acc_wr[ch_ppc];
    }

L24: MERGE_HIST_LOOP:
    for (ap_uint<32> i = 0; i < AEC_HISTSIZE; i++) {
#pragma HLS pipeline II=auto{_PIPE_L24}
#pragma HLS unroll factor=auto{_UNROLL_L24}
        uint32_t value = 0;
L25: MERGE_HIST_NPPC_UNROLL:
        for (ap_uint<5> p = 0; p < XF_NPIXPERCYCLE(NPC); p++) {
#pragma HLS pipeline II=auto{_PIPE_L25}
#pragma HLS unroll factor=auto{_UNROLL_L25}
            value += tmp_hist[p][i];
        }
        hist[i] = value;
    }
}

template <int SRC_T,
          int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int USE_URAM = 0,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int WORDWIDTH,
          int SRC_TC,
          int PLANES,
          int AEC_HISTSIZE>
void xFHistogramKernel_multi(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                             xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& src2,
                             uint32_t hist[AEC_HISTSIZE],
                             float p,
                             float inputMin,
                             float inputMax,
                             float outputMin,
                             float outputMax,
                             int slc_id) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    /* L26: */
#pragma HLS array_partition variable=hist type=auto{_ARRAY_T_L26} factor=auto{_ARRAY_F_L26} dim=auto{_ARRAY_D_L26}

    const int STEP = XF_DTPIXELDEPTH(SRC_T, NPC);
    int width = src1.cols >> XF_BITSHIFT(NPC);
    int height = src1.rows;
    XF_TNAME(SRC_T, NPC) in_pix, in_pix1, out_pix;
    int writenct = 0;
    int depth = 1;
    int bins = AEC_HISTSIZE;
    int nElements = AEC_HISTSIZE;
    int val;

    if (slc_id == 0) {
L27: INITIALIZE_HIST:
        for (int k = 0; k < AEC_HISTSIZE; k++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=AEC_HISTSIZE max=AEC_HISTSIZE
#pragma HLS pipeline II=auto{_PIPE_L27}
#pragma HLS unroll factor=auto{_UNROLL_L27}
            // clang-format on
            hist[k] = 0;
        }
    }

    // Temporary array used while computing histogram
L28: ap_uint<32> tmp_hist[XF_NPIXPERCYCLE(NPC)][AEC_HISTSIZE];
#pragma HLS array_partition variable=tmp_hist type=auto{_ARRAY_T_L28} factor=auto{_ARRAY_F_L28} dim=auto{_ARRAY_D_L28}

    XF_TNAME(SRC_T, NPC) in_buf, in_buf1, temp_buf;
    bool flag = 0;

L29: HIST_INITIALIZE_LOOP:
    for (ap_uint<32> i = 0; i < AEC_HISTSIZE; i++) {
#pragma HLS pipeline II=auto{_PIPE_L29}
#pragma HLS unroll factor=auto{_UNROLL_L29}
L30:     for (ap_uint<5> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=AEC_HISTSIZE max=AEC_HISTSIZE
#pragma HLS pipeline II=auto{_PIPE_L30}
#pragma HLS unroll factor=auto{_UNROLL_L30}
            // clang-format on
            tmp_hist[j][i] = 0;
        }
    }

L31: static uint32_t old[XF_NPIXPERCYCLE(NPC)] = {};
#pragma HLS array_partition variable=old type=auto{_ARRAY_T_L31} factor=auto{_ARRAY_F_L31} dim=auto{_ARRAY_D_L31}

L32: uint32_t acc_rd[XF_NPIXPERCYCLE(NPC)] = {};
#pragma HLS array_partition variable=acc_rd type=auto{_ARRAY_T_L32} factor=auto{_ARRAY_F_L32} dim=auto{_ARRAY_D_L32}

L33: uint32_t acc_wr[XF_NPIXPERCYCLE(NPC)] = {};
#pragma HLS array_partition variable=acc_wr type=auto{_ARRAY_T_L33} factor=auto{_ARRAY_F_L33} dim=auto{_ARRAY_D_L33}

    int readcnt = 0;
    ap_fixed<STEP + 8, STEP + 2> min_vals = inputMin - 0.5f;
    ap_fixed<STEP + 8, STEP + 2> max_vals = inputMax + 0.5f;
    ap_fixed<STEP + 8, STEP + 2> minValue = min_vals, minValue1 = min_vals;
    ap_fixed<STEP + 8, STEP + 2> maxValue = max_vals, maxValue1 = max_vals;
    ap_fixed<STEP + 8, STEP + 2> interval = ap_fixed<STEP + 8, STEP + 2>(maxValue - minValue) / bins;
    ap_fixed<STEP + 8, 2> internal_inv = ((ap_fixed<STEP + 8, 2>)1 / interval);
    int pos = 0, pos1 = 0;
    int currentBin = 0, currentBin1 = 0;

L34: ROW_LOOP:
    for (int row = 0; row != (height); row++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
#pragma HLS pipeline II=auto{_PIPE_L34}
#pragma HLS unroll factor=auto{_UNROLL_L34}
    // clang-format on
L35: COL_LOOP:
        for (int col = 0; col < (width); col = col + 1) {
// clang-format off
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
#pragma HLS pipeline II=auto{_PIPE_L35}
#pragma HLS unroll factor=auto{_UNROLL_L35}

            // clang-format on
            in_pix = src1.read(row * (width) + col);
            src2.write(row * (width) + col, in_pix);

L36:         for (ap_uint<9> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
#pragma HLS DEPENDENCE variable=tmp_hist array intra false
#pragma HLS pipeline II=auto{_PIPE_L36}
#pragma HLS unroll factor=auto{_UNROLL_L36}
                // clang-format on
                XF_CTUNAME(SRC_T, NPC) val = 0, val1 = 0;
                val = in_pix.range(j * STEP + STEP - 1, j * STEP);

                currentBin = int((val - minValue) * internal_inv);

                if (currentBin == old[j]) {
                    acc_rd[j] = acc_wr[j];
                } else {
                    acc_rd[j] = tmp_hist[j][currentBin];
                }

                tmp_hist[j][old[j]] = acc_wr[j];

                acc_wr[j] = acc_rd[j] + 1;
                old[j] = currentBin;
            }
        }
    }

L37: END_HIST_LOOP:
    for (ap_uint<5> ch_ppc = 0; ch_ppc < XF_NPIXPERCYCLE(NPC); ch_ppc++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=NPC
        #pragma HLS pipeline II=auto{_PIPE_L37}
        #pragma HLS unroll factor=auto{_UNROLL_L37}
        // clang-format on
        uint32_t tmp = old[ch_ppc];
        tmp_hist[ch_ppc][tmp] = acc_wr[ch_ppc];
    }

L38: MERGE_HIST_LOOP:
    for (ap_uint<32> i = 0; i < AEC_HISTSIZE; i++) {
#pragma HLS pipeline II=auto{_PIPE_L38}
#pragma HLS unroll factor=auto{_UNROLL_L38}
        uint32_t value = 0;
L39: MERGE_HIST_NPPC_UNROLL:
        for (ap_uint<5> p = 0; p < XF_NPIXPERCYCLE(NPC); p++) {
#pragma HLS pipeline II=auto{_PIPE_L39}
#pragma HLS unroll factor=auto{_UNROLL_L39}
            value += tmp_hist[p][i];
        }
        hist[i] += value;
    }
}

template <int SRC_T, int ROWS, int COLS, int NPC = 1, int USE_URAM = 0, int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void calcHist(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src, uint32_t* histogram) {
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8 ");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

L40: uint32_t hist_array[XF_CHANNELS(SRC_T, NPC)][256] = {0};
#pragma HLS array_partition variable=hist_array type=auto{_ARRAY_T_L40} factor=auto{_ARRAY_F_L40} dim=auto{_ARRAY_D_L40}

    uint16_t width = _src.cols >> (XF_BITSHIFT(NPC));
    uint16_t height = _src.rows;

    xFHistogramKernel<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, USE_URAM, XFCVDEPTH_IN, XF_WORDWIDTH(SRC_T, NPC),
                      ((COLS >> (XF_BITSHIFT(NPC))) >> 1), XF_CHANNELS(SRC_T, NPC)>(_src, hist_array, height, width);

L41: for (int i = 0; i < (XF_CHANNELS(SRC_T, NPC)); i++) {
#pragma HLS pipeline II=auto{_PIPE_L41}
#pragma HLS unroll factor=auto{_UNROLL_L41}
L42:     for (int j = 0; j < 256; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS LOOP_FLATTEN off
#pragma HLS pipeline II=auto{_PIPE_L42}
#pragma HLS unroll factor=auto{_UNROLL_L42}
            // clang-format on
            histogram[(i * 256) + j] = hist_array[i][j];
        }
    }
}

} // namespace cv
} // namespace xf
