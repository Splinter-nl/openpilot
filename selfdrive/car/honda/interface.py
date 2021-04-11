#!/usr/bin/env python3
import numpy as np
from cereal import car
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.events import ET
from selfdrive.car.honda.values import CruiseButtons, CAR, HONDA_BOSCH
from selfdrive.car import STD_CARGO_KG, CivicParams, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.controls.lib.longitudinal_planner import _A_CRUISE_MAX_V_FOLLOWING
from selfdrive.car.interfaces import CarInterfaceBase

A_ACC_MAX = max(_A_CRUISE_MAX_V_FOLLOWING)

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName


def compute_gb_honda(accel, speed):
  creep_brake = 0.0
  creep_speed = 2.3
  creep_brake_value = 0.15
  if speed < creep_speed:
    creep_brake = (creep_speed - speed) / creep_speed * creep_brake_value
  return float(accel) / 4.8 - creep_brake


def get_compute_gb_acura():
  # generate a function that takes in [desired_accel, current_speed] -> [-1.0, 1.0]
  # where -1.0 is max brake and 1.0 is max gas
  # see debug/dump_accel_from_fiber.py to see how those parameters were generated
  w0 = np.array([[ 1.22056961, -0.39625418,  0.67952657],
                 [ 1.03691769,  0.78210306, -0.41343188]])
  b0 = np.array([ 0.01536703, -0.14335321, -0.26932889])
  w2 = np.array([[-0.59124422,  0.42899439,  0.38660881],
                 [ 0.79973811,  0.13178682,  0.08550351],
                 [-0.15651935, -0.44360259,  0.76910877]])
  b2 = np.array([ 0.15624429,  0.02294923, -0.0341086 ])
  w4 = np.array([[-0.31521443],
                 [-0.38626176],
                 [ 0.52667892]])
  b4 = np.array([-0.02922216])

  def compute_output(dat, w0, b0, w2, b2, w4, b4):
    m0 = np.dot(dat, w0) + b0
    m0 = leakyrelu(m0, 0.1)
    m2 = np.dot(m0, w2) + b2
    m2 = leakyrelu(m2, 0.1)
    m4 = np.dot(m2, w4) + b4
    return m4

  def leakyrelu(x, alpha):
    return np.maximum(x, alpha * x)

  def _compute_gb_acura(accel, speed):
    # linearly extrap below v1 using v1 and v2 data
    v1 = 5.
    v2 = 10.
    dat = np.array([accel, speed])
    if speed > 5.:
      m4 = compute_output(dat, w0, b0, w2, b2, w4, b4)
    else:
      dat[1] = v1
      m4v1 = compute_output(dat, w0, b0, w2, b2, w4, b4)
      dat[1] = v2
      m4v2 = compute_output(dat, w0, b0, w2, b2, w4, b4)
      m4 = (speed - v1) * (m4v2 - m4v1) / (v2 - v1) + m4v1
    return float(m4)

  return _compute_gb_acura


class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    self.last_enable_pressed = 0
    self.last_enable_sent = 0

    if self.CS.CP.carFingerprint == CAR.ACURA_ILX:
      self.compute_gb = get_compute_gb_acura()
    else:
      self.compute_gb = compute_gb_honda

  @staticmethod
  def compute_gb(accel, speed): # pylint: disable=method-hidden
    raise NotImplementedError

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):

    # normalized max accel. Allowing max accel at low speed causes speed overshoots
    max_accel_bp = [10, 20]    # m/s
    max_accel_v = [0.714, 1.0]  # unit of max accel
    max_accel = interp(v_ego, max_accel_bp, max_accel_v)

    # limit the pcm accel cmd if:
    # - v_ego exceeds v_target, or
    # - a_ego exceeds a_target and v_ego is close to v_target

    eA = a_ego - a_target
    valuesA = [1.0, 0.1]
    bpA = [0.3, 1.1]

    eV = v_ego - v_target
    valuesV = [1.0, 0.1]
    bpV = [0.0, 0.5]

    valuesRangeV = [1., 0.]
    bpRangeV = [-1., 0.]

    # only limit if v_ego is close to v_target
    speedLimiter = interp(eV, bpV, valuesV)
    accelLimiter = max(interp(eA, bpA, valuesA), interp(eV, bpRangeV, valuesRangeV))

    # accelOverride is more or less the max throttle allowed to pcm: usually set to a constant
    # unless aTargetMax is very high and then we scale with it; this help in quicker restart

    return float(max(max_accel, a_target / A_ACC_MAX)) * min(speedLimiter, accelLimiter)

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=[]):  # pylint: disable=dangerous-default-value
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "honda"

    if candidate in HONDA_BOSCH:
      ret.safetyModel = car.CarParams.SafetyModel.hondaBoschHarness
      ret.enableCamera = True
      ret.radarOffCan = True
      ret.openpilotLongitudinalControl = False
    else:
      ret.safetyModel = car.CarParams.SafetyModel.hondaNidec
      ret.enableCamera = True
      ret.enableGasInterceptor = 0x201 in fingerprint[0]
      ret.openpilotLongitudinalControl = ret.enableCamera

    cloudlog.warning("ECU Camera Simulated: %r", ret.enableCamera)
    cloudlog.warning("ECU Gas Interceptor: %r", ret.enableGasInterceptor)

    ret.enableCruise = not ret.enableGasInterceptor
    ret.communityFeature = ret.enableGasInterceptor

    # Certain Hondas have an extra steering sensor at the bottom of the steering rack,
    # which improves controls quality as it removes the steering column torsion from feedback.
    # Tire stiffness factor fictitiously lower if it includes the steering column torsion effect.
    # For modeling details, see p.198-200 in "The Science of Vehicle Dynamics (2014), M. Guiggiani"
    ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0], [0]]
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.lateralTuning.pid.kf = 0.00004  # conservative feed-forward, base 0.00006

    eps_modified = True
    for fw in car_fw:
      if fw.ecu == "eps" and b"," in fw.fwVersion:
        eps_modified = True

    if candidate in (CAR.CIVIC_BOSCH, CAR.CIVIC_BOSCH_DIESEL):
      stop_and_go = True
      ret.mass = CivicParams.MASS
      ret.wheelbase = CivicParams.WHEELBASE
      ret.centerToFront = CivicParams.CENTER_TO_FRONT
      ret.steerRatio = 15.38  # 10.93 is end-to-end spec
      ret.steerActuatorDelay = 0.04 # base 0.1
      ret.steerRateCost = 0.4 # base 0.5
      ret.steerLimitTimer = 0.8 # base 0.8
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2564, 10000], [0, 2564, 3840]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.12], [0.04]] # base [[0.15], [0.05]]
      tire_stiffness_factor = 1.
      ret.longitudinalTuning.kpBP = [0., 5., 35.]
      ret.longitudinalTuning.kpV = [1.2, 0.8, 0.5]
      ret.longitudinalTuning.kiBP = [0., 35.]
      ret.longitudinalTuning.kiV = [0.18, 0.12]

    else:
      raise ValueError("unsupported car %s" % candidate)

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter. Otherwise, add 0.5 mph margin to not
    # conflict with PCM acc
    ret.minEnableSpeed = -1. if (stop_and_go or ret.enableGasInterceptor) else 25.5 * CV.MPH_TO_MS

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    ret.gasMaxBP = [0.]  # m/s
    ret.gasMaxV = [0.6] if ret.enableGasInterceptor else [0.]  # max gas allowed
    ret.brakeMaxBP = [5., 20.]  # m/s
    ret.brakeMaxV = [1., 0.8]   # max brake allowed

    ret.stoppingControl = True
    ret.startAccel = 0.5

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    # ******************* do can recv *******************
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)
    if self.cp_body:
      self.cp_body.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_cam, self.cp_body)

    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid and (self.cp_body is None or self.cp_body.can_valid)
    ret.yawRate = self.VM.yaw_rate(ret.steeringAngleDeg * CV.DEG_TO_RAD, ret.vEgo)
    # FIXME: read sendcan for brakelights
    brakelights_threshold = 0.02 if self.CS.CP.carFingerprint == CAR.CIVIC else 0.1
    ret.brakeLights = bool(self.CS.brake_switch or
                           c.actuators.brake > brakelights_threshold)

    buttonEvents = []

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.unknown
      if self.CS.cruise_buttons != 0:
        be.pressed = True
        but = self.CS.cruise_buttons
      else:
        be.pressed = False
        but = self.CS.prev_cruise_buttons
      if but == CruiseButtons.RES_ACCEL:
        be.type = ButtonType.accelCruise
      elif but == CruiseButtons.DECEL_SET:
        be.type = ButtonType.decelCruise
      elif but == CruiseButtons.CANCEL:
        be.type = ButtonType.cancel
      elif but == CruiseButtons.MAIN:
        be.type = ButtonType.altButton3
      buttonEvents.append(be)

    if self.CS.cruise_setting != self.CS.prev_cruise_setting:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.unknown
      if self.CS.cruise_setting != 0:
        be.pressed = True
        but = self.CS.cruise_setting
      else:
        be.pressed = False
        but = self.CS.prev_cruise_setting
      if but == 1:
        be.type = ButtonType.altButton1
      # TODO: more buttons?
      buttonEvents.append(be)
    ret.buttonEvents = buttonEvents

    # events
    events = self.create_common_events(ret, pcm_enable=False)
    if self.CS.brake_error:
      events.add(EventName.brakeUnavailable)
    if self.CS.brake_hold and self.CS.CP.openpilotLongitudinalControl:
      events.add(EventName.brakeHold)
    if self.CS.park_brake:
      events.add(EventName.parkBrake)

    if self.CP.enableCruise and ret.vEgo < self.CP.minEnableSpeed:
      events.add(EventName.belowEngageSpeed)

    # it can happen that car cruise disables while comma system is enabled: need to
    # keep braking if needed or if the speed is very low
    if self.CP.enableCruise and not ret.cruiseState.enabled \
       and (c.actuators.brake <= 0. or not self.CP.openpilotLongitudinalControl):
      # non loud alert if cruise disables below 25mph as expected (+ a little margin)
      if ret.vEgo < self.CP.minEnableSpeed + 2.:
        events.add(EventName.speedTooLow)
      else:
        events.add(EventName.cruiseDisabled)
    if self.CS.CP.minEnableSpeed > 0 and ret.vEgo < 0.001:
      events.add(EventName.manualRestart)

    cur_time = self.frame * DT_CTRL
    enable_pressed = False
    # handle button presses
    for b in ret.buttonEvents:

      # do enable on both accel and decel buttons
      if b.type in [ButtonType.accelCruise, ButtonType.decelCruise] and not b.pressed:
        self.last_enable_pressed = cur_time
        enable_pressed = True

      # do disable on button down
      if b.type == "cancel" and b.pressed:
        events.add(EventName.buttonCancel)

    if self.CP.enableCruise:
      # KEEP THIS EVENT LAST! send enable event if button is pressed and there are
      # NO_ENTRY events, so controlsd will display alerts. Also not send enable events
      # too close in time, so a no_entry will not be followed by another one.
      # TODO: button press should be the only thing that triggers enable
      if ((cur_time - self.last_enable_pressed) < 0.2 and
          (cur_time - self.last_enable_sent) > 0.2 and
          ret.cruiseState.enabled) or \
         (enable_pressed and events.any(ET.NO_ENTRY)):
        events.add(EventName.buttonEnable)
        self.last_enable_sent = cur_time
    elif enable_pressed:
      events.add(EventName.buttonEnable)

    ret.events = events.to_msg()

    self.CS.out = ret.as_reader()
    return self.CS.out

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):
    if c.hudControl.speedVisible:
      hud_v_cruise = c.hudControl.setSpeed * CV.MS_TO_KPH
    else:
      hud_v_cruise = 255

    pcm_accel = int(clip(c.cruiseControl.accelOverride, 0, 1) * 0xc6)

    can_sends = self.CC.update(c.enabled, self.CS, self.frame,
                               c.actuators,
                               c.cruiseControl.speedOverride,
                               c.cruiseControl.override,
                               c.cruiseControl.cancel,
                               pcm_accel,
                               hud_v_cruise,
                               c.hudControl.lanesVisible,
                               hud_show_car=c.hudControl.leadVisible,
                               hud_alert=c.hudControl.visualAlert)

    self.frame += 1
    return can_sends
