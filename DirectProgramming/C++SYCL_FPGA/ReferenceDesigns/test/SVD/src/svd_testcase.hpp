#ifndef __SVD_TESTCASE__
#define __SVD_TESTCASE__

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "svd.hpp"
#include "svd_helper.hpp"


template<typename T,
         unsigned rows_A,
         unsigned cols_A,
         bool is_complex = false
         >
struct SVDTestcase
{
    std::vector<std::vector<T>> input_A;
    std::vector<T> output_S;
    T S_error;
    float S_error_r;
    T A_error;
    float A_error_r;
    T U_orthogonal_error;
    T V_orthogonal_error;

    SVDTestcase(std::vector<std::vector<T>> A, 
                std::vector<T> S) :
        input_A(A), output_S(S) {}

    std::vector<T> col_major_A()
    {
        std::vector<T> flat_A;

        for (int col = 0; col < cols_A; col ++)
        {
            for (int row = 0; row < rows_A; row ++)
            {
                flat_A.push_back(input_A[row][col]);
            }
        }
        return flat_A;
    }

    std::vector<T> extract_singular_value(std::vector<T> mat_S)
    {
        std::vector<T> singular_value;
        for (int i = 0; i < rows_A; i ++) {
            singular_value.push_back(mat_S[i * rows_A + i]);
        }
        return singular_value;
    }

    T compare_S(std::vector<T> input_vec)
    {   
        T max_diff = 0.0; 
        float max_ratio = 0.0;
        // current SVD don't reorder singuler value
        std::sort(std::begin(input_vec), std::end(input_vec), std::greater<>());
        for (int i = 0; i < output_S.size(); i ++) {
            T cur_diff = abs(input_vec[i] - output_S[i]);
            if (cur_diff > max_diff) max_diff = cur_diff;
            if ((cur_diff / abs(output_S[i])) >  max_ratio) max_ratio = cur_diff / abs(output_S[i]);
        }
        S_error = max_diff;
        S_error_r = max_ratio;
        return max_diff;
    }

    T check_USV(std::vector<T> flat_A, std::vector<T> flat_U, 
                std::vector<T> flat_S, std::vector<T> flat_V)
    {
        // U @ S
        std::vector<T> US(rows_A * cols_A, 0);
        svd_testbench_tool::soft_matmult<T>(flat_U, rows_A, rows_A, 
                                        flat_S, rows_A, cols_A,
                                        US);
        // transpose to get Vt
        std::vector<T> Vt(cols_A * cols_A, 0);
        svd_testbench_tool::soft_transpose<T>(flat_V, cols_A, cols_A, Vt);
        // US @ Vt
        std::vector<T> USV(rows_A * cols_A, 0);
        svd_testbench_tool::soft_matmult<T>(US, rows_A, cols_A, 
                                        Vt, cols_A, cols_A,
                                        USV);
        // svd_testbench_tool::print_matrix<T>(USV, rows_A, cols_A);
        T max_diff = 0.0;
        float max_ratio = 0.0;
        for (int i = 0; i < (rows_A * cols_A); i ++)
        {
            T cur_diff = abs(USV[i] - flat_A[i]);
            if (cur_diff > max_diff) max_diff = cur_diff;
            if ((cur_diff / abs(flat_A[i])) >  max_ratio) max_ratio = cur_diff / abs(flat_A[i]);
        }
        A_error = max_diff;
        A_error_r = max_ratio;
        return max_diff;
    }

    T check_orthogonal(std::vector<T> flat_mat, 
                        unsigned rows, unsigned cols)
    {
        // check mat @ mat transpose == identity
        std::vector<T> mat_t(cols * rows, 0);
        std::vector<T> mat_i(rows * rows, 0);
        svd_testbench_tool::soft_transpose<T>(flat_mat, rows, cols, mat_t);
        svd_testbench_tool::soft_matmult<T>(flat_mat, rows, cols, 
                                        mat_t, cols, rows,
                                        mat_i);
        T max_diff = 0.0;
        for ( int i = 0; i < (rows * rows); i ++)
        {
            int cur_col = int(i / rows);
            int cur_row = i % rows;
            T cur_diff = 0.0;
            if (cur_row == cur_col) {
                cur_diff = abs(mat_i[i] - 1.0);
            }
            else {
                cur_diff = abs(mat_i[i]);
            }

            if (cur_diff > max_diff) max_diff = cur_diff;
        }
        return max_diff;
    }

    T run_test(sycl::queue q, bool print_result=false, bool timed=false)
    {
        std::vector<T> flat_A = col_major_A();
        std::vector<T> flat_U(rows_A * rows_A);
        std::vector<T> flat_S(rows_A * cols_A);
        std::vector<T> flat_V(cols_A * cols_A);

        auto start = std::chrono::high_resolution_clock::now();
        if (timed) start = std::chrono::high_resolution_clock::now();
        singularValueDecomposition<rows_A, cols_A, false, T>
            (flat_A, flat_U, flat_S, flat_V, q);
        if (timed) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Kernel runtime: "
                << duration.count() << " milliseconds" << std::endl;
        }
        compare_S(extract_singular_value(flat_S));

        A_error = check_USV(flat_A, flat_U, flat_S, flat_V);
        U_orthogonal_error = check_orthogonal(flat_U, rows_A, rows_A);
        V_orthogonal_error = check_orthogonal(flat_V, cols_A, cols_A);

        if (print_result) {
            std::cout << "S:\n"; 
            // print_matrix(extract_singular_value(flat_S), 1, rows_A);
            svd_testbench_tool::print_matrix<T>(flat_S, rows_A, cols_A, true);
            std::cout << "V:\n"; 
            svd_testbench_tool::print_matrix<T>(flat_V, cols_A, cols_A, true);
            std::cout << "U:\n";
            svd_testbench_tool::print_matrix<T>(flat_U, rows_A, rows_A, true);
        }
        return std::max({S_error, A_error, U_orthogonal_error, V_orthogonal_error});
    }

    void print_result()
    {
        std::cout << "Singular value delta: " << S_error << "(" << S_error_r * 100 << "%)" << std::endl;
        std::cout << "Decomposition delta (A = USVt): " << A_error << "(" << A_error_r * 100 << "%)"<< std::endl;
        std::cout << "U orthogonal delta: " << U_orthogonal_error << std::endl;
        std::cout << "V orthogonal delta: " << V_orthogonal_error << std::endl;
    }
};


SVDTestcase<float, 4, 4> small_4x4_trivial(
    std::vector<std::vector<float>> {
        {0.47084338, 0.99594452, 0.47982739, 0.69202168},
        {0.45148837, 0.72836647, 0.64691844, 0.62442883},
        {0.80974833, 0.82555856, 0.30709051, 0.58230306},
        {0.97898197, 0.98520343, 0.40133633, 0.85319924}
    },
    std::vector<float> (
        {2.79495619, 0.44521050, 0.19458290, 0.07948970}
    )
);

SVDTestcase<float, 5, 4> small_5x4_trivial(
    std::vector<std::vector<float>> {
        {0.18951157, 0.68602816, 0.41020029, 0.06032407},
        {0.90243932, 0.13550672, 0.66982882, 0.90253133},
        {0.62553737, 0.99174058, 0.13948975, 0.24745720},
        {0.52667188, 0.12236896, 0.16383614, 0.87902079},
        {0.69672670, 0.14061496, 0.46443579, 0.21936906}
    },
    std::vector<float> (
        {2.15850446, 1.03645533, 0.51339127, 0.30648523}
    )
);

SVDTestcase<float, 4, 5> small_4x5_trivial(
    std::vector<std::vector<float>> {
        {0.69338269, 0.04813966, 0.46807229, 0.33419638, 0.23766854},
        {0.78685005, 0.02949695, 0.94957251, 0.38040955, 0.26970604},
        {0.40872266, 0.42573641, 0.03341264, 0.80447107, 0.14009654},
        {0.87201349, 0.20768288, 0.02503632, 0.66540070, 0.29889724}
    },
    std::vector<float> (
        {2.01743975, 0.87469350, 0.35592941, 0.01246613}
    )
);

SVDTestcase<float, 5, 5> large_5x5_AAt(
    std::vector<std::vector<float>> {
        {196.93499638, 678.45745323, 41.76398579, 165.11409121, 631.89765684},
        {113.37863309, 640.00588122, 868.41620368, 868.96521706, 798.41512213},
        {112.24065970, 687.31725469, 801.71656737, 529.09112139, 428.69311582},
        {754.30396850, 482.82957868, 964.82932930, 867.78226481, 713.70480360},
        {858.00297222, 546.99627721, 740.17378626, 780.88472890, 525.05854018}
    },
    std::vector<float> (
        {3092.17464461, 742.77736366, 593.13092899, 277.60457140, 100.08122958}
    )
);

SVDTestcase<float, 5, 5> small_5x5_F(
    std::vector<std::vector<float>> {
        {0.20275263, 0.04173628, 0.40914067, 0.22228170, 0.82768945},
        {0.20515865, 0.68339494, 0.84414345, 0.12941136, 0.03079066},
        {0.60075713, 0.50273746, 0.78630443, 0.59745787, 0.96850610},
        {0.05675729, 0.63544371, 0.97175501, 0.83885902, 0.29345985},
        {0.77632860, 0.95373265, 0.02879162, 0.95667381, 0.54350833}
    },
    std::vector<float> (
        {2.73999751, 1.02241687, 0.89771150, 0.45666241, 0.09233437}
    )
);

SVDTestcase<float, 6, 5> small_6x5_trivial(
    std::vector<std::vector<float>> {
        {0.60241971, 0.57116204, 0.47683828, 0.26318978, 0.84867509},
        {0.19401475, 0.13060422, 0.39190480, 0.62264985, 0.13164666},
        {0.58354783, 0.99733612, 0.37338498, 0.90658951, 0.47285952},
        {0.56478716, 0.16367478, 0.16417126, 0.41208619, 0.52308584},
        {0.02959032, 0.55890369, 0.53035001, 0.46645708, 0.18075489},
        {0.14643737, 0.36800709, 0.37059426, 0.43164694, 0.57537324}
    },
    std::vector<float> (
        {2.51254960, 0.73850945, 0.47394027, 0.43978866, 0.18881570}
    )
);

SVDTestcase<float, 16, 16> small_16x16_trivial(
    std::vector<std::vector<float>> {
        {0.46350788, 0.81147927, 0.82402582, 0.52257054, 0.59489931, 0.20281449, 0.89595606, 0.58335784, 0.62698680, 0.26275262, 0.74503903, 0.61687417, 0.70328695, 0.30743574, 0.08359028, 0.54334430},
        {0.17854902, 0.10560214, 0.09556397, 0.83391408, 0.93836232, 0.41449330, 0.23074051, 0.22898071, 0.88505398, 0.14477652, 0.49675291, 0.63008341, 0.97753441, 0.23143129, 0.35207622, 0.42497195},
        {0.71994576, 0.11364118, 0.75480370, 0.73220213, 0.84452363, 0.37339272, 0.05621195, 0.65613329, 0.37085795, 0.98419566, 0.14584462, 0.67203998, 0.89393865, 0.85484663, 0.80084825, 0.18941152},
        {0.06488283, 0.21963394, 0.61087088, 0.18874082, 0.71578069, 0.96544540, 0.86384018, 0.91428148, 0.65696661, 0.56748906, 0.68564688, 0.88459356, 0.47523717, 0.79514803, 0.40029808, 0.64209968},
        {0.13317229, 0.81090083, 0.20491509, 0.02368686, 0.82788231, 0.70970873, 0.80833654, 0.96131032, 0.21002413, 0.50641662, 0.31430415, 0.66940176, 0.50451502, 0.73912806, 0.23340388, 0.39022118},
        {0.91572222, 0.32385699, 0.34985974, 0.26214315, 0.01131212, 0.58745392, 0.59987312, 0.64624868, 0.81815052, 0.34163114, 0.99980925, 0.44020213, 0.92216322, 0.19900963, 0.17168076, 0.29794636},
        {0.64547695, 0.89317027, 0.96799449, 0.89856139, 0.31185726, 0.24970117, 0.21513068, 0.89342021, 0.26453942, 0.82143977, 0.25056306, 0.92174708, 0.37745030, 0.35040385, 0.70717672, 0.15146735},
        {0.74228943, 0.78782278, 0.33518245, 0.66823071, 0.14683950, 0.92737776, 0.52653284, 0.14880999, 0.62903218, 0.19964019, 0.54469979, 0.91714209, 0.32460450, 0.08310940, 0.06242663, 0.64249292},
        {0.60883882, 0.50480910, 0.12065972, 0.56900023, 0.33462892, 0.27529851, 0.34315817, 0.53861889, 0.95219629, 0.72230646, 0.86586586, 0.18214830, 0.12915793, 0.57658868, 0.31769888, 0.80294644},
        {0.62364114, 0.25319306, 0.96607966, 0.29128522, 0.75114610, 0.19387555, 0.77391073, 0.89666549, 0.22578486, 0.07759731, 0.40184569, 0.34729276, 0.05065087, 0.85584611, 0.66652579, 0.69655566},
        {0.07798688, 0.77047311, 0.73041636, 0.49383314, 0.11657051, 0.98629649, 0.23090973, 0.74746902, 0.50650100, 0.77358623, 0.84327093, 0.92614728, 0.25353581, 0.75407990, 0.52219490, 0.18705468},
        {0.89213956, 0.88208387, 0.26599840, 0.10550437, 0.51224066, 0.24760013, 0.93106100, 0.18118898, 0.27586143, 0.60646642, 0.70827865, 0.02611148, 0.33935102, 0.45206633, 0.91502295, 0.32610613},
        {0.58530154, 0.22651190, 0.18530080, 0.47734975, 0.69836154, 0.45655683, 0.94820405, 0.71757436, 0.45817830, 0.36090766, 0.93556936, 0.08111286, 0.07966913, 0.60549204, 0.09099894, 0.50185035},
        {0.09671361, 0.35527415, 0.56647001, 0.62111656, 0.17643992, 0.56762009, 0.54393842, 0.92533844, 0.88883615, 0.59417360, 0.09618461, 0.93746541, 0.80054707, 0.32302736, 0.61466426, 0.25437477},
        {0.39743601, 0.43466088, 0.30793011, 0.65884849, 0.81010188, 0.50286563, 0.03967808, 0.24409661, 0.13024388, 0.57317384, 0.14838221, 0.43824248, 0.13947673, 0.33656911, 0.50749376, 0.20080640},
        {0.66424896, 0.45615638, 0.67030826, 0.59712185, 0.74556401, 0.98187031, 0.93864993, 0.88351444, 0.07581938, 0.13592234, 0.20874681, 0.40295596, 0.14583079, 0.36499721, 0.71858545, 0.97918135}
    },
    std::vector<float> (
        {8.24283846, 2.04466224, 1.89265631, 1.59914402, 1.50882543, 1.31846793,
        1.18488904, 1.07799154, 0.91188354, 0.7502934,  0.66594347, 0.52143184,
        0.38370047, 0.3086788,  0.20646569, 0.12279293}
    )
);

// SVDTestcase<float, 8, 7> small_8x7_trivial(
//     std::vector<std::vector<float>> {
//         {9.94276513, 1.59418708, 5.31570427, 2.94699892, 2.29633415, 9.37988807, 8.25643245},
//         {7.81594749, 1.21988457, 0.30268154, 5.81121829, 2.97090789, 4.54063621, 9.46520084},
//         {0.74409819, 9.06879738, 6.39989633, 4.86288086, 0.82151103, 6.50149874, 2.38621947},
//         {7.08608763, 6.56053656, 7.22496268, 2.74371415, 3.40240259, 0.64436602, 0.56465921},
//         {2.82707925, 4.46653124, 9.58671086, 8.79447370, 8.11066938, 3.93688174, 0.58116739},
//         {3.36719360, 4.09415584, 0.92482172, 0.44672662, 2.95142948, 6.11604914, 8.06165813},
//         {3.40863242, 9.05156828, 6.37608842, 3.85044907, 8.35446212, 1.77950092, 5.11185653},
//         {5.29533017, 7.01724814, 5.05005671, 7.69724011, 9.57228067, 4.78429019, 6.58473354}
//     },
//     std::vector<float> (
//         {37.92440275, 15.14685087, 8.42309268, 7.93628259, 6.67266843, 4.00485461, 1.22240010}
//     )
// );

SVDTestcase<float, 8, 7> small_8x7_trivial(
    std::vector<std::vector<float>> {
        {8.76260202, 0.81417924, 8.11057592, 6.09861721, 1.16945558, 7.14839054, 2.59521331},
        {5.44698794, 8.77256268, 9.91432134, 8.25487798, 1.97363324, 1.89532970, 8.33827919},
        {5.08441743, 1.75202020, 3.33778067, 0.63361451, 1.52199570, 1.24156224, 5.70908663},
        {6.46296038, 6.53813124, 9.44048016, 2.59326866, 6.23851206, 1.97112360, 0.46819856},
        {7.49739756, 6.15580471, 9.69197449, 0.44904051, 5.58367157, 1.12969807, 8.62452559},
        {5.36318497, 8.06629730, 6.80937533, 3.42879733, 0.35662368, 3.83916105, 4.85440952},
        {7.75598987, 6.29906745, 2.93344563, 8.20448999, 0.32866946, 5.49268734, 2.22140202},
        {5.30055998, 3.08502860, 9.16451937, 0.13376870, 5.52183665, 7.17266690, 2.92410590}
    },
    std::vector<float> (
        {38.65450260, 11.11997526, 9.82668675, 6.70742472, 4.21300971, 4.12512634, 2.09643874}
    )
);


#endif // __SVD_TESTCASE__