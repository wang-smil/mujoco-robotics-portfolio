"""Reward v2: track mean body twist, preserve normal periodic COM sway.

Reference commands are interpolated because the real grid has neither vy=0
nor wz=0. Actor101 is unchanged. Critic212 repurposes the two unused airtime
slots and redundant phase-index slot for the three filter states; raw twist
and sin/cos phase remain observable. Checkpoints are explicit transfer starts.
"""
import jax
import jax.numpy as jp
import numpy as np

from playground.open_duck_mini_v2 import locomotion
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

REWARD_VERSION='locomotion-gait-v2'


def default_config():
    cfg=locomotion.default_config('forward')
    cfg.twist_filter_tau=.2
    cfg.reset_pose_scale=.2
    return cfg


def filter_twist(previous,measurement,dt,tau):
    alpha=1.-jp.exp(-dt/tau)
    return previous+alpha*(measurement-previous)


def tracking_twist(command,filtered,measurement):
    # Standing should be quiet, not just zero-mean shaking.
    return jp.where(jp.linalg.norm(command[:3])<.01,measurement,filtered)


def coherent_task_info(info,actor_phase):
    """Brax resets data/obs only. Restore task histories on the next input.

    Reset phase is [0,0]; every stepped phase is [cos,sin] with unit norm.
    The mismatch therefore identifies a wrapper reset without depending on
    wrapper-owned done/truncation flags, which are cleared before env.step.
    """
    reset=jp.any(actor_phase!=info['imitation_phase'])
    result=dict(info)
    for key,initial in info['_episode_reset'].items():
        if key!='rng':  # Keep the evolving noise/command RNG stream.
            result[key]=jax.tree_util.tree_map(lambda value,first:jp.where(reset,first,value),info[key],initial)
    return result


def patch_filtered_observation(obs,twist):
    critic=obs['privileged_state'].at[167:169].set(twist[:2])
    critic=critic.at[209].set(twist[2])
    return dict(state=obs['state'],privileged_state=critic)


class InterpolatedReference:
    """Trilinear interpolation of frames at the task's integer control phases."""
    def __init__(self,source):
        self._source=source
        self._axes=[jp.asarray(axis) for axis in (source.dxs,source.dys,source.dthetas)]
        if any(len(axis)<2 for axis in self._axes):
            raise ValueError('Reference interpolation needs at least two grid nodes per axis')
        n=source.nb_steps_in_period
        coefficients=source.data_array
        phases=jp.arange(n,dtype=jp.float32)/n
        sample=source.sample_polynomial
        build=jax.jit(jax.vmap(lambda c:jax.vmap(lambda t:sample(t,c))(phases)))
        flat=coefficients.reshape((-1,coefficients.shape[-2],coefficients.shape[-1]))
        # Evaluate before interpolating to avoid mixing large polynomial
        # coefficients and introducing additional cancellation near phase1.
        self._frames=build(flat).reshape((*coefficients.shape[:3],n,coefficients.shape[-2]))
        if not np.isfinite(np.asarray(self._frames)).all():
            raise ValueError('Reference contains nonfinite frames')

    def __getattr__(self,name):
        return getattr(self._source,name)

    def get_reference_motion(self,dx,dy,dtheta,i):
        weights=[]
        for axis,value in zip(self._axes,(dx,dy,dtheta)):
            value=jp.clip(value,axis[0],axis[-1])
            upper=jp.clip(jp.sum(value>=axis),1,len(axis)-1)
            lower=upper-1
            fraction=(value-axis[lower])/(axis[upper]-axis[lower])
            weights.append(((lower,1.-fraction),(upper,fraction)))
        phase=jp.asarray(i % self.nb_steps_in_period,dtype=jp.int32)
        result=jp.zeros(self._frames.shape[-1])
        for x,wx in weights[0]:
            for y,wy in weights[1]:
                for z,wz in weights[2]:
                    result=result+(wx*wy*wz)*self._frames[x,y,z,phase]
        return result


class GaitLocomotion(locomotion.Locomotion):
    def __init__(self,task='flat_terrain',config=None,config_overrides=None):
        super().__init__(task=task,config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def _post_init(self):
        super()._post_init()
        self.PRM=InterpolatedReference(self.PRM)

    def reset(self,rng):
        state=super().reset(rng)
        # The original multiplicative +/-50% joint perturbation can start in
        # severe penetration/asymmetry. First-stage curriculum uses +/-10%
        # around the same home pose, retaining random xy/yaw/base velocity.
        qpos=state.data.qpos.at[7:].set(self._init_q[7:]+
            self._config.reset_pose_scale*(state.data.qpos[7:]-self._init_q[7:]))
        data=mjx_env.init(self.mjx_model,qpos=qpos,qvel=state.data.qvel,ctrl=state.data.ctrl)
        info=dict(state.info)
        info['imu_history']=jp.zeros_like(info['imu_history'])
        contact=jp.array([geoms_colliding(data,geom,self._floor_geom_id) for geom in self._feet_geom_id])
        state=state.replace(data=data,info=info,obs=self._get_obs(data,info,contact))
        state.info['filtered_twist']=jp.concatenate([self.get_local_linvel(state.data)[:2],
                                                   self.get_gyro(state.data)[2:3]])
        state.info['_episode_reset']=dict(state.info)
        return state.replace(obs=patch_filtered_observation(state.obs,state.info['filtered_twist']))

    def step(self,state,action):
        state=state.replace(info=coherent_task_info(state.info,state.obs['state'][-2:]))
        state=super().step(state,action)
        return state.replace(obs=patch_filtered_observation(state.obs,state.info['filtered_twist']))

    def _get_reward(self,data,action,info,metrics,done,first_contact,contact):
        measurement=jp.concatenate([self.get_local_linvel(data)[:2],self.get_gyro(data)[2:3]])
        filtered=filter_twist(info['filtered_twist'],measurement,self.dt,self._config.twist_filter_tau)
        tracked=tracking_twist(info['command'],filtered,measurement)
        terms=super()._get_reward(data,action,info,metrics,done,first_contact,contact)
        linear,angular=locomotion.tracking_rewards(info['command'],
            jp.concatenate([tracked[:2],jp.zeros(1)]),jp.array([0.,0.,tracked[2]]),
            self._config.reward_config.linear_sigma,self._config.reward_config.angular_sigma)
        terms.update(tracking_lin_vel=linear,tracking_ang_vel=angular)
        terms,invalid=locomotion.sanitize_terms(terms,info['invalid_transition'])
        info['filtered_twist']=filtered
        info['invalid_transition']=invalid
        info['signed_reward']=locomotion.signed_reward(terms,self._config.reward_config.scales,self.dt)
        return terms
