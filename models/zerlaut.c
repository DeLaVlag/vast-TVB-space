// -*- coding: utf-8 -*-
//
//
// TheVirtualBrain-Scientific Package. This package holds all simulators, and
// analysers necessary to run brain-simulations. You can use it stand alone or
// in conjunction with TheVirtualBrain-Framework Package. See content of the
// documentation-folder for more details. See also http://www.thevirtualbrain.org
//
// (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
//
// This program is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this
// program.  If not, see <http://www.gnu.org/licenses/>.
//
//
//   CITATION:
// When using The Virtual Brain for scientific publications, please cite it as follows:
//
//  Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
//  Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
//      The Virtual Brain: a simulator of primate brain network dynamics.
//   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
//
//
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h> // for printf
#define PI_2 (2 * M_PI_F)
#define PI M_PI_F
#define INF INFINITY
#define SQRT2 1.414213562

// buffer length defaults to the argument to the integrate kernel
// but if it's known at compile time, it can be provided which allows
// compiler to change i%n to i&(n-1) if n is a power of two.
#ifndef NH
#define NH nh
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#include <curand_kernel.h>
#include <curand.h>
#include <stdbool.h>


//float b_e = 60.0;
float E_L_e = -63.0;
float E_L_i = -65.0;
float T = 19.0;

// regular global constants
const float g_L = 10.0;
const float C_m = 200.0;
//const float a_e = 0.0;
const float b_i = 0.0;
const float a_i = 0.0;
//const float tau_w_e = 500.0;
const float tau_w_i = 1.0;
const float E_e = 0.0;
const float E_i = -80.0;
const float Q_e = 1.5;
const float Q_i = 5.0;
const float tau_e = 5.0;
// tau_i is fitted for showcase1x
const float tau_i = 5.0;
const float N_tot = 10000;
const float p_connect_e = 0.05;
const float p_connect_i = 0.05;
const float g = 0.2;

#include "models/TFparameters.h"

const float tau_OU = 5;
//const float weight_noise = 1e-4;
const float K_ext_e = 400;
const float K_ext_i = 0;

#include "models/zerlaut.h"

__device__ float wrap_it_EI(float EI)
{
    float EIdim[] = {0.0, 3000};
    if (EI < EIdim[0]) EI = EIdim[0];
    else if (EI > EIdim[1]) EI = EIdim[1];

    return EI;
}

__global__ void zerlaut(

        // config
        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_work_items,
        float dt, float conduct_speed, int my_rank, float * __restrict__ weights, float * __restrict__ lengths,
        float * __restrict__ params_pwi, // pwi: per work item
        // state
        float * __restrict__ state_pwi,
        // outputs
        float * __restrict__ tavg_pwi
        )
{
    // work id & size
//    const unsigned int id = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int size = n_work_items;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
//    const unsigned int id = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    // 2d grid with 2d blocks
    const unsigned int id = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // 2d grid with 1d blocks
//    const unsigned int id = blockId * blockDim.x + threadIdx.x;

#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * 6 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // only threat those ID that have a corresponding parameters combination
    if (id >= size) return;

    // unpack params
    // Goldman parameters/grid search demo
    const float global_coupling = params(0);
    const float b_e = params(1);
    const float weight_noise = params(2);
    const float global_speed = params(3);
    const float tau_w_e = params(4);
    const float a_e = params(5);
    const float external_input_ex_ex = params(6);
    const float external_input_ex_in = params(7);
    const float external_input_in_ex = params(8);
    const float external_input_in_in = params(9);

    const float E_L_e = -63;
    const float E_L_i = -65;
    const float T = 19;


//    const float alpha_g = 0.002;
    const float beta_g = 0;
    const float gamma_g = -0.05;
    const float delta_g = 0;


    // for function pointers
    float mu_V;
    float sigma_V;
    float T_V;

    // coupling constants, coupling itself is hardcoded in kernel
    const float c_a = 1;

    // coupling parameters
    float c_pop0 = 0.0;

    // derived parameters
    const float rec_n = 1 / n_node;
    const float rec_speed_dt = 1.0f / global_speed / (dt);
//    const float nsig = sqrt(dt) * sqrt(2.0 * 1e-3);
//    const float local_coupling = 0.0;

    // the dynamic derived variables declarations
    float E_input_excitatory = 0.0;
    float E_input_inhibitory = 0.0;
    float I_input_excitatory = 0.0;
    float I_input_inhibitory = 0.0;

    float N_e = N_tot * (1-g);
    float N_i = N_tot * g;

    float E = 0.0;
    float I = 0.0;
    float C_ee = 0.0;
    float C_ei = 0.0;
    float C_ii = 0.0;
    float W_e = 100.0;
    float W_i = 0.0;
    float noise = 0.0;

    float dE = 0.0;
    float dI = 0.0;
    float dC_ee = 0.0;
    float dC_ei = 0.0;
    float dC_ii = 0.0;
    float dW_e = 0.0;
    float dW_i = 0.0;
    float dnoise = 0.0;

    unsigned int dij_i = 0;
    float dij = 0.0;
    float wij = 0.0;

    float V_j = 0.0;

    float df = 1e-7;
    float powdf = powf((df*1e3), 2);
    float lc_E = 0;
    float lc_I = 0;

    // Print parameters
    if (my_rank == 0 && id==0 && i_step==0)
    {
        printf("   INFO  Model parameters:\n");
        printf("   INFO  global_coupling =      %.1f \n", global_coupling);
        printf("   INFO  b_e =                  %.1f \n", b_e);
        printf("   INFO  weight_noise =         %.6f \n", weight_noise);
        printf("   INFO  g_L =                  %.1f \n", g_L);
        printf("   INFO  C_m =                  %.1f \n", C_m);
        printf("   INFO  a_e =                  %.1f \n", a_e);
        printf("   INFO  b_i =                  %.1f \n", b_i);
        printf("   INFO  a_i =                  %.1f \n", a_i);
        printf("   INFO  tau_w_e =              %.1f \n", tau_w_e);
        printf("   INFO  tau_w_i =              %.1f \n", tau_w_i);
        printf("   INFO  E_e =                  %.1f \n", E_e);
        printf("   INFO  E_i =                  %.1f \n", E_i);
        printf("   INFO  Q_e =                  %.1f \n", Q_e);
        printf("   INFO  Q_i =                  %.1f \n", Q_i);
        printf("   INFO  tau_e =                %.1f \n", tau_e);
        printf("   INFO  tau_i =                %.1f \n", tau_i);
        printf("   INFO  N_tot =                %.1f \n", N_tot);
        printf("   INFO  p_connect_e =          %.2f \n", p_connect_e);
        printf("   INFO  p_connect_i =          %.2f \n", p_connect_i);
        printf("   INFO  g =                    %.1f \n", g);
        printf("   INFO  external_input_ex_ex = %f \n", external_input_ex_ex);
        printf("   INFO  external_input_ex_in = %f \n", external_input_ex_in);
        printf("   INFO  external_input_in_ex = %f \n", external_input_in_ex);
        printf("   INFO  external_input_in_in = %f \n", external_input_in_in);
        printf("   INFO  tau_OU =               %.1f \n", tau_OU);
        printf("   INFO  weight_noise =         %.4f \n", weight_noise);
        printf("   INFO  K_ext_e =              %.1f \n", K_ext_e);
        printf("   INFO  K_ext_i =              %.1f \n", K_ext_i);
        printf("   INFO  global_speed =         %.1f \n", global_speed);
        printf("   INFO  E_input_excitatory =   %.1f \n", E_input_excitatory);
        printf("   INFO  E_input_inhibitory =   %.1f \n", E_input_inhibitory);
        printf("   INFO  I_input_excitatory =   %.1f \n", I_input_excitatory);
        printf("   INFO  I_input_inhibitory =   %.1f \n", I_input_inhibitory);
        printf("   INFO  N_e =                  %.1f \n", N_e);
        printf("   INFO  N_i =                  %.1f \n", N_i);
        printf("   INFO  P_e [%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f]\n", P_e0, P_e1, P_e2, P_e3, P_e4, P_e5, P_e6, P_e7, P_e8, P_e9);
        printf("   INFO  P_i [%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f]\n", P_i0, P_i1, P_i2, P_i3, P_i4, P_i5, P_i6, P_i7, P_i8, P_i9);
        printf("\n\n");

    }

    curandState crndst;
    curand_init(id + (unsigned int) clock64(), 0, 0, &crndst);

    // This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
    {

        if (i_step == 0){
            state(i_step, i_node) = 0.0f;
        }
    }

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (int i_node = 0; i_node < n_node; i_node++)
        {

            c_pop0 = 0.0f;

            if (t == (i_step)){
                tavg(i_node + 0 * n_node) = 0;
                tavg(i_node + 1 * n_node) = 0;
//                tavg(i_node + 2 * n_node) = 0;
//                tavg(i_node + 3 * n_node) = 0;
//                tavg(i_node + 4 * n_node) = 0;
//                tavg(i_node + 5 * n_node) = 0;
//                tavg(i_node + 6 * n_node) = 0;
            }

            // check if the state !=<0
            E = state((t) % nh, i_node + 0 * n_node);
            I = state((t) % nh, i_node + 1 * n_node);
            C_ee = state((t) % nh, i_node + 2 * n_node);
            C_ei = state((t) % nh, i_node + 3 * n_node);
            C_ii = state((t) % nh, i_node + 4 * n_node);
//            if (t!=0) W_e = state((t) % nh, i_node + 5 * n_node);
            if (t==0){W_e = 100;}
            else {W_e = state((t) % nh, i_node + 5 * n_node);}

//            W_i = state((t) % nh, i_node + 6 * n_node);
            noise = state((t) % nh, i_node + 6 * n_node);

            E = wrap_it_EI(E);
            I = wrap_it_EI(I);

            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement. /
            unsigned int i_n = i_node * n_node;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;

                // Get the delay between node i and node j
                dij = lengths[i_n + j_node] * rec_speed_dt;
                dij = dij + 0.5;
                dij_i = (int)dij;

                //***// Get the state of node j which is delayed by dij
                V_j = state(((t - dij_i + nh) % nh), j_node + 0 * n_node);

                // Sum it all together using the coupling function.
                c_pop0 += wij * V_j;
            } // j_node */

            // global coupling handling, rec_n used to scale nodes
            c_pop0 *= global_coupling;
            // the dynamic derived variables declarations
            E_input_excitatory = c_pop0+lc_E+external_input_ex_ex + weight_noise * noise;
            E_input_inhibitory = c_pop0+lc_E+external_input_in_ex + weight_noise * noise;

            // The conditional variables
            if (E_input_excitatory < 0.0) E_input_excitatory = 0.0;
            if (E_input_inhibitory < 0.0) E_input_inhibitory = 0.0;
            I_input_excitatory = lc_I+external_input_ex_in;
            I_input_inhibitory = lc_I+external_input_in_in;

            // Transfer function of excitatory and inhibitory neurons
            float _TF_e = TF_excitatory(E, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e);
            float _TF_i = TF_inhibitory(E, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i);

//            printf("istep %d _TF_e %f \n", i_step, _TF_e);

            float _diff_fe_e = (TF_excitatory(E+df, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                               -TF_excitatory(E-df, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                               )/(2*df*1e3);

            float _diff_fe_i = (TF_inhibitory(E+df, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                               -TF_inhibitory(E-df, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                               )/(2*df*1e3);

            float _diff_fi_e = (TF_excitatory(E, I+df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                               -TF_excitatory(E, I-df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                               )/(2*df*1e3);

            float _diff_fi_i = (TF_inhibitory(E, I+df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                               -TF_inhibitory(E, I-df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                               )/(2*df*1e3);

            float _diff2_fe_fe_e = (TF_excitatory(E+df, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                   -2*_TF_e
                                   +TF_excitatory(E-df, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                   )/powdf;

            float _diff2_fe_fe_i = (TF_inhibitory(E+df, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                   -2*_TF_i
                                   +TF_inhibitory(E-df, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                   )/powdf;

            // diff of different
            float _diff2_fe_fi = (TF_excitatory(E+df, I+df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                 -TF_excitatory(E+df, I-df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                 -TF_excitatory(E-df, I+df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                 +TF_excitatory(E-df, I-df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                 )/(4*powdf);

            float _diff2_fi_fe = (TF_inhibitory(E+df, I+df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                 -TF_inhibitory(E+df, I-df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                 -TF_inhibitory(E-df, I+df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                 +TF_inhibitory(E-df, I-df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                 )/(4*powdf);

            float _diff2_fi_fi_e = (TF_excitatory(E, I+df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                   -2*_TF_e
                                   +TF_excitatory(E, I-df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                   )/powdf;

            float _diff2_fi_fi_i = (TF_inhibitory(E, I+df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                   -2*_TF_i
                                   +TF_inhibitory(E, I-df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                   )/powdf;


            // Integrate with forward euler
            // equation is inspired from github of Zerlaut :
            // https://github.com/yzerlaut/notebook_papers/blob/master/modeling_mesoscopic_dynamics/mean_field/master_equation.py

            // Excitatory firing rate derivation
            dE = dt * ((_TF_e - E
                + .5*C_ee*_diff2_fe_fe_e
                + C_ei*_diff2_fe_fi
                + .5*C_ii*_diff2_fi_fi_e
                    )/T);

            // Inhibitory firing rate derivation
            dI = dt * ((_TF_i - I
                + .5*C_ee*_diff2_fe_fe_i
                + C_ei*_diff2_fi_fe
                + .5*C_ii*_diff2_fi_fi_i
                    )/T);

            // Covariance excitatory-excitatory derivation
            dC_ee = dt * ((_TF_e*(1./T-_TF_e)/N_e
                + powf((_TF_e-E), 2)
                + 2.*C_ee*_diff_fe_e
                + 2.*C_ei*_diff_fi_i
                - 2.*C_ee
                    )/T);

            // Covariance excitatory-inhibitory or inhibitory-excitatory derivation
            dC_ei = dt * (((_TF_e-E)*(_TF_i-I)
                + C_ee*_diff_fe_e
                + C_ei*_diff_fe_i
                + C_ei*_diff_fi_e
                + C_ii*_diff_fi_i
                - 2.*C_ei
                    )/T);

            // Covariance inhibitory-inhibitory derivation
            dC_ii = dt * ((_TF_i*(1./T-_TF_i)/N_i
                + powf((_TF_i-I), 2)
                + 2.*C_ii*_diff_fi_i
                + 2.*C_ei*_diff_fe_e
                - 2.*C_ii
                    )/T);

            // Adaptation excitatory
            get_fluct_regime_vars(E, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e,
                                    &mu_V, &sigma_V, &T_V);
            dW_e = dt * (-W_e/tau_w_e+b_e*E+a_e*(mu_V-E_L_e)/tau_w_e);

//            get_fluct_regime_vars(E, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i,
//                                    &mu_V, &sigma_V, &T_V);
//            dW_i = dt * (-W_i/tau_w_i+b_i*I+a_i*(mu_V-E_L_i)/tau_w_i);

            dnoise = dt * (-noise/tau_OU);

            E += dE;
            I += dI;
            C_ee += dC_ee;
            C_ei += dC_ei;
            C_ii += dC_ii;
            W_e += dW_e;
            W_i += dW_i;
            noise += dnoise + (curand_normal(&crndst));

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = E;
            state((t + 1) % nh, i_node + 1 * n_node) = I;
            state((t + 1) % nh, i_node + 2 * n_node) = C_ee;
            state((t + 1) % nh, i_node + 3 * n_node) = C_ei;
            state((t + 1) % nh, i_node + 4 * n_node) = C_ii;
            state((t + 1) % nh, i_node + 5 * n_node) = W_e;
//            state((t + 1) % nh, i_node + 6 * n_node) = W_i;
            state((t + 1) % nh, i_node + 6 * n_node) = noise;

            // Update the observable
            tavg(i_node + 0 * n_node) += E/n_step;
//            tavg(i_node + 1 * n_node) += W_e/n_step;
            tavg(i_node + 1 * n_node) += I/n_step;
//            tavg(i_node + 2 * n_node) += C_ee/n_step;
//            tavg(i_node + 3 * n_node) += C_ei/n_step;
//            tavg(i_node + 4 * n_node) += C_ii/n_step;
//            tavg(i_node + 5 * n_node) += W_e/n_step;
//            tavg(i_node + 6 * n_node) += W_i/n_step;

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate

// defaults from Stefan 2007, cf tvb/analyzers/fmri_balloon.py
#define TAU_S 0.65f
#define TAU_F 0.41f
#define TAU_O 0.98f
#define ALPHA 0.32f
#define TE 0.04f
#define V0 4.0f
#define E0 0.4f
#define EPSILON 0.5f
#define NU_0 40.3f
#define R_0 25.0f

#define RECIP_TAU_S (1.0f / TAU_S)
#define RECIP_TAU_F (1.0f / TAU_F)
#define RECIP_TAU_O (1.0f / TAU_O)
#define RECIP_ALPHA (1.0f / ALPHA)
#define RECIP_E0 (1.0f / E0)

// "derived parameters"
#define k1 (4.3f * NU_0 * E0 * TE)
#define k2 (EPSILON * R_0 * E0 * TE)
#define k3 (1.0f - EPSILON)

__global__ void bold_update(int n_node, float dt,
                      // bold.shape = (4, n_nodes, n_threads)
            float * __restrict__ bold_state,
                      // nrl.shape = (n_nodes, n_threads)
            float * __restrict__ neural_state,
                      // out.shape = (n_nodes, n_threads)
            float * __restrict__ out)
{
    const unsigned int it = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int nt = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

    int var_stride = n_node * nt;
    for (int i_node=0; i_node < n_node; i_node++)
    {
        float *node_bold = bold_state + i_node * nt + it;

        float s = node_bold[0 * var_stride];
        float f = node_bold[1 * var_stride];
        float v = node_bold[2 * var_stride];
        float q = node_bold[3 * var_stride];

        float x = neural_state[i_node * nt + it];

        float ds = x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1.0f);
        float df = s;
        float dv = RECIP_TAU_O * (f - pow(v, RECIP_ALPHA));
        float dq = RECIP_TAU_O * (f * (1.0f - pow(1.0f - E0, 1.0f / f))
                * RECIP_E0 - pow(v, RECIP_ALPHA) * (q / v));

        s += dt * ds;
        f += dt * df;
        v += dt * dv;
        q += dt * dq;

        node_bold[0 * var_stride] = s;
        node_bold[1 * var_stride] = f;
        node_bold[2 * var_stride] = v;
        node_bold[3 * var_stride] = q;

        out[i_node * nt + it] = V0 * (    k1 * (1.0f - q    )
                                        + k2 * (1.0f - q / v)
                                        + k3 * (1.0f -     v) );
    } // i_node
} // kernel
