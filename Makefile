SHELL := /bin/zsh

# Path to the Python virtual environment
VENV_PATH ?= .venv
PYTHON ?= $(VENV_PATH)/bin/python


.PHONY: sfig-local-wie-training
sfig-local-wie-training:
	$(PYTHON) run_traj.py --multirun \
		seed=1,2,3,4,5 \
		hydra.job.name=local-wie-training-noprune \
		training_phases='[{name:wie_pretrain,iterations:2160,preset:train_wie_only},{name:main,iterations:4320,preset:train_full}]' \
		rec_every_Nth_iter=20 \
		model.eta_IE=0.1 \
		model.compute_wIE_method=random_scaled_uniform \
		model.w_ie_pruning=False \
		model.w_ie_pruning_thresh=0.01 \
		model.w_ie_pruning_start_iter=720 \


.PHONY: sfig-wEE-and-wII
sfig-wEE-and-wII:
	$(PYTHON) run_traj.py --multirun \
		seed=1,2,3,4,5 \
		hydra.job.name=wEE-and-wII \
		rec_every_Nth_iter=20 \
		model.g_EI=6.0 \
		model.g_XI=1.0 \
		model.g_EE=1.25 \
		model.g_II=0.5 \
		model.eta_EE=0.002 \


.PHONY: sfig-feedback-to-E
sfig-feedback-to-E:
	$(PYTHON) run_traj.py --multirun \
		seed=1,2,3,4,5 \
		hydra.job.name=feedback-to-E \
		rec_every_Nth_iter=20 \
		model.feedback_to_excitatory=True \