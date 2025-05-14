# Run commands to replicate results

Collection of commands that call python scripts to exactly replicate the results from our paper

---

## Figure 4: Trajectory learning

**No Overlap**
```
python run_traj.py rec_activity=False
```
(Set the `rec_activity` flag to `True` to record neuronal activity every `N`-th epoch (controlled by the `rec_every_Nth_iter` argument)

---

## Figure 5: Trajectory learning (overlapping assemblies)

**10% overlap** (Replace 10 with any other percentage)
```
python run_traj.py perc_random_memb=10
```

---

## Figure 6: Fashion-MNIST 

**1-layer network without overlap**
```
python run_static.py +dataset=fmnist model.vf.nb_hidden=1 trainer.clip_params=True
```

**1-layer network with 10% assembly overlap** (replace with other overlap percentage if necessary)

```
python run_static.py +dataset=fmnist model.vf.nb_hidden=1 trainer.clip_params=True model.vf.perc_overlap=10
```

**3-layer network without overlap**
```
python run_static.py +dataset=fmnist model.vf.nb_hidden=3 trainer.clip_params=True model.vf.controller.k_p=0.05 model.vf.controller.k_i=0.1 trainer.norm_fb_weights=True
```

---

## Figure 7: Fear conditioning task

**Train one network initialization on 10 trials of fear conditioning task:**
```
python run_fearcond.py 
```

---

## Figure 8: Motor learning task

**Train one network initialization on 2000 trials of the motor learning task:**
```
python run_motor.py 
```

---

## Supplementary Figure 2: Trajectory learning with random feedback

**Per-assembly random feedback**
```
python run_traj.py model.random_fb_per_ensemble=True
```

**Per-neuron random feedback**
```
python run_traj.py model.random_fb_per_neuron=True
```
