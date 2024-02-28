from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
world_size = comm.Get_size()

import logging
import itertools
import argparse
import pickle

import time

from tvb.simulator.lab import *
from tvb.basic.logger.builder import get_logger

import os.path

try:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule
	import pycuda.gpuarray as gpuarray
except ImportError:
	logging.warning('pycuda not available, rateML driver not usable.')

import tqdm

from analysis.analysis import analysis

from analysis.insert_database import *

np.set_printoptions(precision=2)

here = os.path.dirname(os.path.abspath(__file__))

class Driver_Setup:

	def __init__(self):
		self.args = self.parse_args()

		self.logger = get_logger('tvb.rateML')
		self.logger.setLevel(level='INFO' if self.args.verbose else 'WARNING')

		self.checkargbounds()

		self.dt = self.args.delta_time
		self.connectivity = self.tvb_connectivity(self.args.n_regions)
		# self.weights = self.connectivity.weights
		self.weights = self.connectivity.weights / (np.sum(self.connectivity.weights, axis=0) + 1e-12)

		self.lengths = self.connectivity.tract_lengths
		self.tavg_period = 1
		self.n_inner_steps = int(self.tavg_period / self.dt)

		self.params, self.wi_per_rank = self.setup_params()

		# bufferlength is based on the minimum of the first swept parameter (speed for many tvb models)
		self.n_work_items, self.n_params = self.params.shape
		self.buf_len_ = ((self.lengths / self.args.conduct_speed / self.dt).astype('i').max() + 1)
		self.buf_len = 2 ** np.argwhere(2 ** np.r_[:30] > self.buf_len_)[0][0]  # use next power of

		self.states = self.args.states
		self.exposures = self.args.exposures

		if self.args.gpu_info:
			self.logger.setLevel(level='INFO')
			self.gpu_device_info()
			exit(1)

		if my_rank == 0:
			self.logdata()

	def logdata(self):

		self.logger.info('dt %f', self.dt)
		self.logger.info('s0 %f', self.args.n_sweep_arg0)
		self.logger.info('s1 %f', self.args.n_sweep_arg1)
		# self.logger.info('s2 %f', self.args.n_sweep_arg2)
		# self.logger.info('s3 %f', self.args.n_sweep_arg3)
		# self.logger.info('s4 %f', self.args.n_sweep_arg4)
		self.logger.info('n_nodes %d', self.args.n_regions)
		self.logger.info('weights.shape %s', self.weights.shape)
		self.logger.info('lengths.shape %s', self.lengths.shape)
		self.logger.info('tavg period %s', self.tavg_period)
		self.logger.info('n_inner_steps %s', self.n_inner_steps)
		self.logger.info('params shape %s', self.params.shape)

		self.logger.info('nstep %d', self.args.n_time)
		self.logger.info('n_inner_steps %f', self.n_inner_steps)

		# self.logger.info('single connectome, %d x %d parameter space', self.args.n_sweep_arg0, self.args.n_sweep_arg1)
		self.logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
		self.logger.info('number of states %d', self.states)
		self.logger.info('model %s', self.args.model)
		self.logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
		self.logger.info('memory for states array on GPU %d MiB',
						 (self.buf_len * self.n_work_items * self.states * self.args.n_regions * 4) / 1024 ** 2)


	def checkargbounds(self):

		try:
			assert self.args.n_sweep_arg0 > 0, "Min value for [N_SWEEP_ARG0] is 1"
			assert self.args.n_time > 0, "Minimum number for [-n N_TIME] is 1"
			assert self.args.n_regions > 0, "Min value for  [-tvbn n_regions] for default data set is 68"
			assert self.args.blockszx > 0 and self.args.blockszx <= 32,	"Bounds for [-bx BLOCKSZX] are 0 < value <= 32"
			assert self.args.blockszy > 0 and self.args.blockszy <= 32, "Bounds for [-by BLOCKSZY] are 0 < value <= 32"
			assert self.args.delta_time > 0.0, "Min value for [-dt delta_time] is > 0.0, default is 0.1"
			assert self.args.conduct_speed > 0.0, "Min value for [-sm speeds_min] is > 0.0, default is 3e-3"
			assert self.args.exposures > 0, "Min value for [-x exposures] is 1"
			assert self.args.states > 0, "Min value for [-s states] is 1"
		except AssertionError as e:
			self.logger.error('%s', e)
			raise

	def tvb_connectivity(self, tvbnodes):
		if not self.args.connectome_file:
			sfile="connectivity_"+str(tvbnodes)+".zip"
			white_matter = connectivity.Connectivity.from_file(source_file=sfile)
			if my_rank == 0:
				self.logger.info('connectivity file %s', sfile)
		# white_matter = connectivity.Connectivity.from_file(source_file="paupau.zip")
		# white_matter = connectivity.Connectivity.from_file(source_file= here + "/Connectome/connectivity_zerlaut_68.zip")
		else:
			sfile = here + '/' + self.args.connectome_file
			white_matter = connectivity.Connectivity.from_file(source_file=sfile)
			if my_rank == 0:
				self.logger.info('connectivity file %s', self.args.connectome_file)
		white_matter.configure()


		return white_matter

	def parse_args(self):  # {{{
		parser = argparse.ArgumentParser(description='Run parameter sweep.')

		# for every parameter that needs to be swept, the size can be set
		parser.add_argument('-s0', '--n_sweep_arg0', default=4, help='num grid points for 1st parameter', type=int)
		parser.add_argument('-s1', '--n_sweep_arg1', default=4, help='num grid points for 2st parameter', type=int)
		parser.add_argument('-s2', '--n_sweep_arg2', default=0, help='num grid points for 3st parameter', type=int)
		parser.add_argument('-s3', '--n_sweep_arg3', default=0, help='num grid points for 4st parameter', type=int)
		parser.add_argument('-s4', '--n_sweep_arg4', default=0, help='num grid points for 5st parameter', type=int)
		parser.add_argument('-s5', '--n_sweep_arg5', default=0, help='num grid points for 6st parameter', type=int)
		parser.add_argument('-s6', '--n_sweep_arg6', default=0, help='num grid points for 7st parameter', type=int)
		parser.add_argument('-s7', '--n_sweep_arg7', default=0, help='num grid points for 8st parameter', type=int)
		parser.add_argument('-s8', '--n_sweep_arg8', default=0, help='num grid points for 9st parameter', type=int)
		parser.add_argument('-s9', '--n_sweep_arg9', default=0, help='num grid points for 10st parameter', type=int)
		parser.add_argument('-s10', '--n_sweep_arg10', default=0, help='num grid points for 11st parameter', type=int)
		# parser.add_argument('-s11', '--n_sweep_arg11', default=0, help='num grid points for 12st parameter', type=int)
		parser.add_argument('-n', '--n_time', default=400, help='number of time steps to do', type=int)
		parser.add_argument('-v', '--verbose', default=False, help='increase logging verbosity', action='store_true')
		parser.add_argument('-m', '--model', default='zerlaut_func', help="neural mass model to be used during the simulation")
		parser.add_argument('-s', '--states', default=8, type=int, help="number of states for model")
		parser.add_argument('-x', '--exposures', default=2, type=int, help="number of exposures for model")
		parser.add_argument('-l', '--lineinfo', default=False, help='generate line-number information for device code.', action='store_true')
		parser.add_argument('-bx', '--blockszx', default=32, type=int, help="gpu block size x")
		parser.add_argument('-by', '--blockszy', default=32, type=int, help="gpu block size y")
		parser.add_argument('-val', '--validate', default=False, help="enable validation with refmodels", action='store_true')
		parser.add_argument('-r', '--n_regions', default="68", type=int, help="number of tvb nodes")
		parser.add_argument('-p', '--plot_data', type=int, help="plot res data for selected state")
		parser.add_argument('-w', '--write_data', default=False, help="write output data to file: 'tavg_data", action='store_true')
		parser.add_argument('-g', '--gpu_info', default=False, help="show gpu info", action='store_true')
		parser.add_argument('-dt', '--delta_time', default=0.1, type=float, help="dt for simulation")
		parser.add_argument('-cs', '--conduct_speed', default=3, type=float, help="set conduction speed for temporal buffer")
		parser.add_argument('--procid', default="0", type=int, help="Number of L2L processes(Only when in L2L)")
		parser.add_argument('-gs', '--grid_search', default=False, help="enablel grid search", action='store_true')
		parser.add_argument('-cf', '--connectome_file', default="", help='connectome filename', type=str)
		parser.add_argument('-ff', '--FCD_file', default="", help='FCD filename', type=str)
		parser.add_argument('-lr', '--load_gabaas', default=False, help='increase logging verbosity', action='store_true')

		args = parser.parse_args()
		return args

	# Validatoin params
	# S = 0.4
	# b_e = 120.0
	# E_L_e = -64
	# E_L_i = -64
	# T = 19
	def setup_params(self):
		'''
        This code generates the parameters ranges that need to be set
        '''

		# for mdpi, sanity test
		# slh = [.2, .9, # coupling
		# 	   0.0, 100.0, # b_e
		# 	   -4, -4, # weight_noise
		# 	   3.0, 3.0,# global_speed 1 3 5
		# 	   500.0, 500.,# tau_w_e
		# 	   0.0, 0.0,# a_e
		# 	   0.315*1e-3, 0.315*1e-3,# external_input_ex_in
		# 	   0., 0.,# external_input_in_ex
		# 	   0.315*1e-3, 0.315*1e-3,# external_input_in_in
		# 	   0., 0.# external_input_ex_ex
		# 	   ]

		# for all MDPI
		# slh = [.2, .9,# coupling 8
		# 	   0.0, 100.0,# b_e 8
		# 	   -7, -4,# log(weight_noise) 4
		# 	   1.0, 7.0,# global_speed 4
		# 	   250, 750.,# tau_w_e 4
		# 	   -10.0, 20.0,# a_e 4
		# 	   0.3*1e-3, 0.5*1e-3,# external_input_ex_ex 2
		# 	   0., 0.5*1e-3,# external_input_ex_in 1
		# 	   0.3*1e-3, 0.5*1e-3,# external_input_in_ex 2
		# 	   0., 0.5*1e-3# external_input_in_in 1
		# 	   ]

		# for timeseries plot for pahtlogical fixed point
		slh = [.2, .9,  # coupling 8
			   0.0, 120.0,  # b_e 8
			   -7, -4,  # log(weight_noise) 4
			   1.0, 7.0,  # global_speed 4
			   250, 750.,  # tau_w_e 4
			   -10.0, 20.0,  # a_e 4
			   0.4 *1e-3, 0.5*1e-3,  # external_input_ex_ex 2
			   0., 0.5 * 1e-3,  # external_input_ex_in 1
			   0.4 *1e-3, 0.5*1e-3,  # external_input_in_ex 2
			   0., 0.5 * 1e-3  # external_input_in_in 1
			   ]

		s0 = np.linspace(slh[0], slh[1], self.args.n_sweep_arg0)      # coupling
		s1 = np.linspace(slh[2], slh[3], self.args.n_sweep_arg1)      # b_e
		s2 = np.logspace(slh[4], slh[5], self.args.n_sweep_arg2)      # weight_noise
		s3 = np.linspace(slh[6], slh[7], self.args.n_sweep_arg3)      # global_speed
		s4 = np.linspace(slh[8], slh[9], self.args.n_sweep_arg4)      # tau_w_e
		s5 = np.linspace(slh[10], slh[11], self.args.n_sweep_arg5)    # a_e
		s6 = np.linspace(slh[12], slh[13], self.args.n_sweep_arg6)    # external_input_ex_ex
		s7 = np.linspace(slh[14], slh[15], self.args.n_sweep_arg7)    # external_input_ex_in
		s8 = np.linspace(slh[16], slh[17], self.args.n_sweep_arg8)    # external_input_in_ex
		s9 = np.linspace(slh[18], slh[19], self.args.n_sweep_arg9)    # external_input_in_in
		# s10 = np.linspace(slh[20], slh[21], self.args.n_sweep_arg10)
		# s11 = np.linspace(slh[22], slh[23], self.args.n_sweep_arg11) #
		#
		# params = itertools.product(s0, s1, s2, s3, s4)
		params = itertools.product(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9)#, s10#, s11)
		# params = itertools.product(s0, s1)
		params = np.array([vals for vals in params], np.float32)

		if my_rank == 0:
			self.logger.info('linspace(%.1f, %.1f, %d)', slh[0], slh[1], self.args.n_sweep_arg0)
			self.logger.info('linspace(%.1f, %.1f, %d)', slh[2], slh[3], self.args.n_sweep_arg1)
			self.logger.info('linspace(%.4f, %.4f, %d)', slh[4], slh[5], self.args.n_sweep_arg2)
			self.logger.info('linspace(%.1f, %.1f, %d)', slh[6], slh[7], self.args.n_sweep_arg3)
			self.logger.info('linspace(%.1f, %.1f, %d)', slh[8], slh[9], self.args.n_sweep_arg4)

		# mpi stuff
		print(params.shape)
		wi_total, wi_one_size = params.shape
		wi_per_rank = int(wi_total / world_size)
		wi_remaining = wi_total%world_size
		params_pr = params.reshape((world_size, wi_per_rank, wi_one_size))

		if my_rank == 0:
			self.logger.info('remaining wi %f', wi_remaining)
			self.logger.info('params shape per world %s', params.shape)
			self.logger.info('params shape per rank %s', params_pr.shape)

		# only return the params per rank
		return params_pr[my_rank, :, :].squeeze(), wi_per_rank


	def gpu_device_info(self):
		'''
		Get GPU device information
		TODO use this information to give user GPU setting suggestions
		'''
		dev = drv.Device(0)
		print('\n')
		self.logger.info('GPU = %s', dev.name())
		self.logger.info('TOTAL AVAIL MEMORY: %d MiB', dev.total_memory()/1024/1024)

		# get device information
		att = {'MAX_THREADS_PER_BLOCK': [],
			   'MAX_BLOCK_DIM_X': [],
			   'MAX_BLOCK_DIM_Y': [],
			   'MAX_BLOCK_DIM_Z': [],
			   'MAX_GRID_DIM_X': [],
			   'MAX_GRID_DIM_Y': [],
			   'MAX_GRID_DIM_Z': [],
			   'TOTAL_CONSTANT_MEMORY': [],
			   'WARP_SIZE': [],
			   # 'MAX_PITCH': [],
			   'CLOCK_RATE': [],
			   'TEXTURE_ALIGNMENT': [],
			   # 'GPU_OVERLAP': [],
			   'MULTIPROCESSOR_COUNT': [],
			   'SHARED_MEMORY_PER_BLOCK': [],
			   'MAX_SHARED_MEMORY_PER_BLOCK': [],
			   'REGISTERS_PER_BLOCK': [],
			   'MAX_REGISTERS_PER_BLOCK': []}

		for key in att:
			getstring = 'drv.device_attribute.' + key
			# att[key].append(eval(getstring))
			self.logger.info(key + ': %s', dev.get_attribute(eval(getstring)))

class Driver_Execute(Driver_Setup):

	def __init__(self, ds):
		self.args = ds.args
		self.set_CUDAmodel_dir()
		self.weights, self.lengths, self.params = ds.weights, ds.lengths, ds.params
		self.buf_len, self.states, self.n_work_items = ds.buf_len, ds.states, ds.n_work_items
		self.n_inner_steps, self.n_params, self.dt = ds.n_inner_steps, ds.n_params, ds.dt
		self.exposures, self.logger = ds.exposures, ds.logger
		self.conduct_speed = ds.args.conduct_speed
		self.connectivity = ds.connectivity
		self.wi_per_rank = ds.wi_per_rank

	def set_CUDAmodel_dir(self):
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
								 "../generatedModels", self.args.model.lower() + '.c')

	def set_CUDA_ref_model_dir(self):
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
								 "../generatedModels/cuda_refs", self.args.model.lower() + '.c')

	def make_kernel(self, source_file, warp_size, args, lineinfo=True, nh='nh'):

		try:
			with open(source_file, 'r') as fd:
				source = fd.read()
				source = source.replace('M_PI_F', '%ff' % (np.pi, ))
				opts = ['--ptxas-options=-v', '-maxrregcount=32']
				# if lineinfo:
				opts.append('-lineinfo')
				opts.append('-g')
				opts.append('-DWARP_SIZE=%d' % (warp_size, ))
				opts.append('-DNH=%s' % (nh, ))

				idirs = [here, here]
				if my_rank == 0:
					self.logger.info('nvcc options %r', opts)

				try:
					network_module = SourceModule(
							source, options=opts, include_dirs=idirs,
							no_extern_c=True,
							keep=False,)
				except drv.CompileError as e:
					self.logger.error('Compilation failure \n %s', e)
					exit(1)

				# generic func signature creation
				mod_func = '_Z7zerlautjjjjjffiPfS_S_S_S_'

				step_fn = network_module.get_function(mod_func)

			with open(here + '/models/covar.c', 'r') as fd:
				source = fd.read()
				opts = ['-ftz=true']  # for faster rsqrtf in corr
				opts.append('-DWARP_SIZE=%d' % (warp_size,))
				# opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x,))
				covar_module = SourceModule(source, options=opts)
				covar_fn = covar_module.get_function('update_cov')
				cov_corr_fn = covar_module.get_function('cov_to_corr')

			# with open(here + '/models/balloon.c', 'r') as fd:
			# 	source = fd.read()
			# 	opts = []
			# 	opts.append('-DWARP_SIZE=%d' % (warp_size,))
			# 	# opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x,))
			# 	bold_module = SourceModule(source, options=opts)
			# 	bold_fn = bold_module.get_function('bold_update')

		except FileNotFoundError as e:
			self.logger.error('%s.\n  Generated model filename should match model on cmdline', e)
			exit(1)


		return step_fn, covar_fn, cov_corr_fn
		# for + bold
		# return step_fn, bold_fn, covar_fn, cov_corr_fn

	def cf(self, array):#{{{
		# coerce possibly mixed-stride, double precision array to C-order single precision
		return array.astype(dtype='f', order='C', copy=True)#}}}

	def nbytes(self, data):#{{{
		# count total bytes used in all data arrays
		nbytes = 0
		for name, array in data.items():
			nbytes += array.nbytes
		return nbytes#}}}

	def make_gpu_data(self, data):#{{{
		# put data onto gpu
		gpu_data = {}
		for name, array in data.items():
			try:
				gpu_data[name] = gpuarray.to_gpu(self.cf(array))
			except drv.MemoryError as e:
				self.gpu_mem_info()
				self.logger.error(
					'%s.\n\t Please check the parameter dimensions, %d parameters are too large for this GPU 0',
					e, self.params.size)
				exit(1)
		return gpu_data#}}}

	def release_gpumem(self, gpu_data):
		for name, array in gpu_data.items():
			try:
				gpu_data[name].gpudata.free()
			except drv.MemoryError as e:
				self.logger.error('%s.\n\t Freeing mem error', e)
				exit(1)

	def gpu_mem_info(self):

		cmd = "nvidia-smi -q -d MEMORY"#,UTILIZATION"
		os.system(cmd)  # returns the exit code in unix

	def run_simulation(self):

		# setup data#{{{
		data = { 'weights': self.weights, 'lengths': self.lengths, 'params': self.params.T}
		base_shape = self.n_work_items,
		for name, shape in dict(
			tavg0=(self.exposures, self.args.n_regions,),
			tavg1=(self.exposures, self.args.n_regions,),
			state=(self.buf_len, self.states * self.args.n_regions),
			# bold_state=(4, self.args.n_regions),
			# bold=(self.args.n_regions,),
			covar_means=(2 * self.args.n_regions,),
			covar_cov=(self.args.n_regions, self.args.n_regions,),
			corr=(self.args.n_regions, self.args.n_regions,),
			).items():
			# memory error exception for compute device
			try:
				data[name] = np.zeros(shape + base_shape, 'f')
			except MemoryError as e:
				self.logger.error('%s.\n\t Please check the parameter dimensions %d x %d, they are to large '
							 'for this compute device',
							 e, self.args.n_sweep_arg0, self.args.n_sweep_arg1)
				exit(1)
		#  set all but first to 1
		# data['bold_state'][1:] = 1.0

		gpu_data = self.make_gpu_data(data)

		# setup CUDA stuff#{{{
		# step_fn, bold_fn, covar_fn, cov_corr_fn = self.make_kernel(
		step_fn, covar_fn, cov_corr_fn = self.make_kernel(
			# source_file=self.args.filename,
			source_file=here + '/models/zerlaut.c',
			warp_size=32,
			# block_dim_x=self.args.n_sweep_arg0,
			# ext_options=preproccesor_defines,
			# caching=args.caching,
			args=self.args,
			lineinfo=self.args.lineinfo,
			nh=self.buf_len,
			)

		# setup simulation#
		tic = time.time()

		n_streams = 32
		streams = [drv.Stream() for i in range(n_streams)]
		events = [drv.Event() for i in range(n_streams)]
		tavg_unpinned = []
		bold_unpinned = []

		try:
			tavg = drv.pagelocked_zeros((n_streams,) + data['tavg0'].shape, dtype=np.float32)
			# bold = drv.pagelocked_zeros((n_streams,) + data['bold'].shape, dtype=np.float32)
		except drv.MemoryError as e:
			self.logger.error(
				'%s.\n\t Please check the parameter dimensions, %d parameters are too large for this GPU 1',
				e, self.params.size)
			exit(1)

		# determine optimal grid recursively
		def dog(fgd):
			maxgd, mingd = max(fgd), min(fgd)
			maxpos = fgd.index(max(fgd))
			if (maxgd - 1) * mingd * bx * by >= nwi:
				fgd[maxpos] = fgd[maxpos] - 1
				dog(fgd)
			else:
				return fgd

		# n_sweep_arg0 scales griddim.x, n_sweep_arg1 scales griddim.y
		# form an optimal grid recursively
		bx, by = self.args.blockszx, self.args.blockszy
		nwi = self.n_work_items
		rootnwi = int(np.ceil(np.sqrt(nwi)))
		gridx = int(np.ceil(rootnwi / bx))
		gridy = int(np.ceil(rootnwi / by))

		final_block_dim = bx, by, 1

		fgd = [gridx, gridy]
		dog(fgd)
		final_grid_dim = fgd[0], fgd[1]
		# final_grid_dim = 4, 4

		assert gridx * gridy * bx * by >= nwi

		if my_rank == 0:
			self.logger.info('work items %r', self.n_work_items)
			self.logger.info('history shape %r', gpu_data['state'].shape)
			self.logger.info('gpu_data_tavg %s', gpu_data['tavg0'].shape)
			# self.logger.info('gpu_data_bold %s', gpu_data['bold'].shape)
			# self.logger.info('gpu_data_bold_state %s', gpu_data['bold_state'].shape)
			self.logger.info('gpu_data_covm %s', gpu_data['covar_means'].shape)
			self.logger.info('gpu_data_covc %s', gpu_data['covar_cov'].shape)
			self.logger.info('gpu_data_corr %s', gpu_data['corr'].shape)
			self.logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))
			self.logger.info('final block dim %r', final_block_dim)
			self.logger.info('final grid dim %r', final_grid_dim)
			self.gpu_mem_info() if self.args.verbose else None

		# run simulation#{{{
		nstep = self.args.n_time

		if my_rank == 0:
			tqdm_iterator = tqdm.trange(nstep)
		else:
			tqdm_iterator = range(nstep)

		try:
			# for i in tqdm.trange(nstep, file=sys.stdout):
			for i in tqdm_iterator:

				try:
					event = events[i % n_streams]
					stream = streams[i % n_streams]

					if i > 0:
						stream.wait_for_event(events[(i - 1) % n_streams])

					step_fn(np.uintc(i * self.n_inner_steps), np.uintc(self.args.n_regions), np.uintc(self.buf_len),
							np.uintc(self.n_inner_steps), np.uintc(self.n_work_items),
							np.float32(self.dt), np.float32(self.conduct_speed), np.uintc(my_rank),
							gpu_data['weights'], gpu_data['lengths'], gpu_data['params'],
							gpu_data['state'],
							gpu_data['tavg%d' % (i%2,)],
							block=final_block_dim, grid=final_grid_dim)

					event.record(streams[i % n_streams])
				except drv.LaunchError as e:
					self.logger.error('%s', e)
					exit(1)

				tavgk = 'tavg%d' % ((i + 1) % 2,)
				# bold_fn(np.uintc(self.args.n_regions),
				# 		# BOLD model dt is in s, requires 1e-3
				# 		np.float32(self.dt * self.n_inner_steps * 1e-3),
				# 		np.uintc(self.n_work_items),
				# 		gpu_data['bold_state'], gpu_data[tavgk][0,:,:], gpu_data['bold'],
				# 		block=final_block_dim, grid=final_grid_dim, stream=stream)

				# print('gpudatashape', gpu_data['tavg%d' % (i%2,)].shape)

				if i >= (nstep // 2):
					i_time = i - nstep // 2
					# update_cov (covar_cov is output, tavgk and covar_means are input)
					covar_fn(np.uintc(i_time), np.uintc(self.args.n_regions), np.uintc(self.n_work_items),
							 gpu_data['covar_cov'], gpu_data['covar_means'], gpu_data[tavgk][0,:,:],
							 # gpu_data['corr'], gpu_data['covar_means'], gpu_data[tavgk],
							 block=final_block_dim, grid=final_grid_dim,
							 stream=stream)

				# async wrt. other streams & host, but not this stream.
				if i >= n_streams:
					stream.synchronize()
					tavg_unpinned.append(tavg[i % n_streams].copy())
					# bold_unpinned.append(bold[i % n_streams].copy())

				drv.memcpy_dtoh_async(tavg[i % n_streams], gpu_data[tavgk].ptr, stream=stream)
				# drv.memcpy_dtoh_async(bold[i % n_streams], gpu_data['bold'].ptr, stream=stream)

				if i == (nstep - 1):
				# cov_to_corr(covar_cov is input, and corr output)
					cov_corr_fn(np.uintc(nstep // 2), np.uintc(self.args.n_regions), np.uintc(self.n_work_items),
								gpu_data['covar_cov'], gpu_data['corr'],
								# block=(couplings.size, 1, 1), grid=(speeds.size, 1), stream=stream)
								block=final_block_dim, grid=final_grid_dim,
								stream=stream)

			# recover uncopied data from pinned buffer
			if nstep > n_streams:
				for i in range(nstep % n_streams, n_streams):
					stream.synchronize()
					tavg_unpinned.append(tavg[i].copy())
					# bold_unpinned.append(bold[i].copy())

			for i in range(nstep % n_streams):
				stream.synchronize()
				tavg_unpinned.append(tavg[i].copy())
				# bold_unpinned.append(bold[i].copy())

			corr = gpu_data['corr'].get()

		except drv.LogicError as e:
			self.logger.error('%s. Check the number of states of the model or '
						 'GPU block shape settings blockdim.x/y %r, griddim %r.',
						 e, final_block_dim, final_grid_dim)
			exit(1)
		except drv.RuntimeError as e:
			self.logger.error('%s', e)
			exit(1)


		# self.logger.info('kernel finish..')
		# release pinned memory
		tavg = np.array(tavg_unpinned)
		# bold = np.array(bold_unpinned)

		# also release gpu_data
		self.release_gpumem(gpu_data)

		if my_rank == 0:
			self.logger.info('kernel finished')

		bold = 0
		return tavg, corr, bold

	def make_TS_DictList(self, tavg, cut_transient, bestindex):
		from datetime import datetime
		from collections import OrderedDict
		timestring = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

		# coupling
		# b_e
		# weight_noise
		# T
		# global_speed
		# tau_w_e
		# a_e
		# external_input_ex_ex
		# external_input_ex_in
		# external_input_in_ex
		# external_input_in_in

		TSdictlist = []
		parameter_name = ['g', 'be', 'wNoise', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in']

		for ind, pars in zip(bestindex, self.params[bestindex]):
			my_dict = OrderedDict(zip(parameter_name, pars))
			my_dict['time'] = timestring
			my_dict.move_to_end('time', last=False)
			my_dict['TS'] = tavg[cut_transient:, :, :, ind]
			TSdictlist.append(my_dict)

		return TSdictlist

	def calc_corrcoef_SC(self, corr):
		# calculate correlation between SC and simulated FC. SC is the weights of TVB simulation.
		SC = self.connectivity.weights / self.connectivity.weights.max()
		ccFCSC = np.zeros(self.n_work_items, 'f')
		for i in range(self.n_work_items):
			ccFCSC[i] = np.corrcoef(corr[:, :, i].ravel(), SC.ravel())[0, 1]

		return ccFCSC

	def calc_corrcoef_selfFC(self, tavg):
		# calculate correlation between results itself.

		ccselfFC = np.zeros(self.n_work_items, 'f')
		for i in range(self.n_work_items):
			ccselfFC[i] = np.corrcoef(
				tavg[:,:,i].ravel())

		return ccselfFC

	def compare_FCD_fromFile(self, ts):
		# calculate correlation between simulated FC and obtained FC, also for phFCD
		# comparefile = open(here + '/' + self.args.FCD_file, 'rb')

		FCD = np.load(here + '/' + self.args.FCD_file, allow_pickle=True)
		self.logger.info('Compare file %s', self.args.FCD_file)

		# FCD = pickle.load(comparefile)
		# comparefile.close()

		if my_rank == 0:
			print('ts', ts.shape)
			print('FCD', FCD.shape)

		if np.isnan(FCD).any():
			print('FCDfromfileNaN?', np.where(np.isnan(FCD))[0], 'in world:', my_rank)

		ccFCFC = np.zeros(self.n_work_items, 'f')
		for i in range(self.n_work_items):
			ccFCFC[i] = np.corrcoef(
				# corr[i, :FCD.shape[0]].ravel(),
				ts[i, :].ravel(),
				FCD[:ts.shape[1]].ravel())[0, 1]

		return ccFCFC

	def comp_phFCD(self, tavg, cut_transient):

		import phFCD
		import BOLDFilters

		# cut_transient //demo overide
		cut_transient = int(self.args.n_time * .2)
		cuttavg = tavg[cut_transient:, :, :, :]
		# make bold
		decimate = 15
		bold_d = cuttavg[::decimate, :, :, :]

		# set bold params
		BOLDFilters.TR = 2.4  # sampling interval. Original was 2.
		BOLDFilters.flp = 0.01  # lowpass frequency of filter. Original was .02
		BOLDFilters.fhi = 0.09
		# phFCD.discardOffset = 10

		# to determine output size
		Tmax = bold_d[:, 0, 0, 0].size
		npattmax = Tmax - (2 * phFCD.discardOffset - 1)  # calculates the size of phfcd vector
		size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)

		# ping = time.time()
		# iterate over all parameter combinations
		n_params = bold_d[0, 0, 0, :].size
		phfcd = np.zeros((n_params, size_kk3), 'f')
		for i in tqdm.trange(n_params):
			phfcd[i] = phFCD.from_fMRI(bold_d[:, 0, :, i].squeeze().T)

		return phfcd


	def runanaly(self, tavg, cut_transient):

		pinganaly = time.time()

		parameter_name = ['g', 'be', 'wNoise', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in']
		path_root = os.path.dirname(os.path.realpath(__file__)) + '/data/'
		database = path_root + "/mGPU_TVB.db"
		table_name = "exploration"
		init_database(database, table_name)

		if my_rank == 0:
			tqdm_analy_iterator = tqdm.trange(self.wi_per_rank)
		else:
			tqdm_analy_iterator = range(self.wi_per_rank)

		results = []
		for i in tqdm_analy_iterator:
			result = analysis(parameter_name, self.params[i, :],
									   tavg[cut_transient:, :, :, i].squeeze(), self.weights, self.lengths)
			if not check_already_analyse_database(database, table_name, parameter_name, self.params[i, :]):
					results.append(result)

		ponganaly = time.time()
		print('time for analy %.2f' % (ponganaly-pinganaly))

		return results

	def run_all(self):

		np.random.seed(79)

		tic = time.time()

		tavg0, corr, bold = self.run_simulation()

		if np.isnan(tavg0).any():
			print('tavgnan?', np.where(np.isnan(tavg0))[0], 'in world:', my_rank)

		# if np.isnan(tavgFC).any():
		# 	print('tavgFCnan?', np.where(np.isnan(tavgFC))[0], 'in world:', my_rank)

		if np.isnan(bold).any():
			print('boldnans?', np.where(np.isnan(bold))[0], 'in world:', my_rank)

		if np.isnan(corr).any():
			print('covanans?', np.where(np.isnan(corr))[0], 'in world:', my_rank)

		toc = time.time()
		elapsed = toc - tic

		if (self.args.validate == True):
			self.compare_with_ref(tavg0)

		# cut transient relative to sim time
		cut_transient = self.args.n_time // 5

		# tavgFC = self.comp_phFCD(tavg0, cut_transient)

		# use the GPU corr
		tavgFC = corr


		if self.args.grid_search:
			# compare with ref pearson file
			ccFC = self.calc_corrcoef_SC(tavgFC)
			ccFC = ccFC * (np.min(tavg0[:, 0, :, :]) < 0.15)

			# compare to reference
			# ccFC = self.compare_FCD_fromFile(tavgFC)

			if my_rank == 0:
				self.logger.info('fitness shape %s', ccFC.shape)
				self.logger.info('max fitness %s', np.max(ccFC))

			# look at finesses gridlike
			fitness_sorting_indices = list(reversed(np.argsort(ccFC, axis=0)))
			sorted_fitness = list(reversed(np.sort(ccFC)))
			sorted_fitness = np.asarray(sorted_fitness).reshape(self.n_work_items, 1)
			paramsfitness = np.c_[self.params[fitness_sorting_indices], sorted_fitness]

			# hardcorded set of parameters used for the input file
			fitness_sorting_indices = np.asarray(fitness_sorting_indices)

			# check individual fitnesses
			FCCOMP = 0
			if FCCOMP == 1:
				# chech individual fitness
				target_params = np.array([0.4, 72., -64., -64., 19.])
				tol = .01 #1e-6
				indices = np.where(np.all(np.isclose(paramsfitness[:, :5], target_params, atol=tol), axis=1))
				fitness = paramsfitness[indices, 5]

				if len(indices) != 0:
					self.logger.info('Target_params is found %s', target_params)
					self.logger.info('	has %s fitnesses', fitness[0])
					self.logger.info('	at position %s', indices[0])
				else:
					self.logger.info('Target_params is not found', target_params)

			# dump only the best one to file
			TSdictlist = self.make_TS_DictList(tavg0, cut_transient, fitness_sorting_indices[:18])

			if np.isnan(ccFC).any():
				print('ccFC?', np.where(np.isnan(ccFC))[0], 'in world:', my_rank)

			analy_result = self.runanaly(tavg0, cut_transient)


		tac = time.time()
		FCelapsed = tac - toc

		if my_rank == 0:
			self.logger.info('Corr shape (bnodes, bnodes, n_params) %s', corr.shape)
			# self.logger.info('Bold shape (simsteps, bnodes, n_params) %s', bold.shape)
			self.logger.info('TS shape (simsteps, states, bnodes, n_params) %s', tavg0.shape)
			self.logger.info('Finished CUDA simulation successfully in: {0:.3f}'.format(elapsed))
			self.logger.info('Finished FC comparison successfully in: {0:.3f}'.format(FCelapsed))
			self.logger.info('and in {0:.3f} M step/s'.format(
				1e-6 * self.args.n_time * self.n_inner_steps * self.n_work_items / elapsed))
			self.logger.info('tavgFC %s', tavgFC.shape)

		return paramsfitness[:10], analy_result, TSdictlist

def write_output(writer):
	# from datetime import datetime
	# timestring = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

	filename = '/data/tavg_b10.npy'# + timestring
	tavg_file = open(here + filename, 'wb')
	pickle.dump(writer, tavg_file)
	tavg_file.close()

if __name__ == '__main__':

	n_runs = 1
	driver_setup = Driver_Setup()
	params_here = driver_setup.params

	# shape is 10 x (1 row params + 1 fitness)
	bestten = np.zeros((10, params_here.shape[1]+1))
	for run in range(n_runs):
		if run == 0:
			bestten, analyres, tsdictlist = Driver_Execute(driver_setup).run_all()
			# print(analyres)
		else:
			b10_analyres = Driver_Execute(driver_setup).run_all()
			bestten = (bestten + b10_analyres[0])/2
			# print(b10_analyres)


	comm.Barrier()
	bestten_world = np.array(comm.gather(bestten, root=0))
	all_analyresul = np.array(comm.gather(analyres, root=0))
	all_tsdictlis = np.array(comm.gather(tsdictlist, root=0))

	if my_rank == 0:
		# save b10 tavgs to file
		write_output(all_tsdictlis)

		# save results to db
		path_root = os.path.dirname(os.path.realpath(__file__)) + '/data/'
		# print(path_root)
		database = path_root + "/mGPU_TVB.db"
		table_name = "exploration"

		print('all_analyresul.shape', all_analyresul.shape)
		for world_res in all_analyresul.T[:]:
			if len(world_res) != 0:
				insert_database(database, table_name, world_res)

		pares = np.array(bestten_world.reshape((world_size*10, params_here.shape[1]+1)))
		# reverse the incides
		indicesrev = np.argsort(pares[:, -1])[::-1]
		print('\n     [    g,    be,    ele,    eli,    T;    fitness]')
		print(pares[indicesrev][:10])

	MPI.Finalize()
