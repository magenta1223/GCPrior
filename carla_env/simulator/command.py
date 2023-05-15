from typing import Union

import carla

CarlaCommand = Union[
    carla.command.ApplyAngularImpulse,
    carla.command.ApplyForce,
    carla.command.ApplyImpulse,
    carla.command.ApplyTargetAngularVelocity,
    carla.command.ApplyTargetVelocity,
    carla.command.ApplyTorque,
    carla.command.ApplyTransform,
    carla.command.ApplyVehicleControl,
    carla.command.ApplyVehiclePhysicsControl,
    carla.command.ApplyWalkerControl,
    carla.command.ApplyWalkerState,
    carla.command.DestroyActor,
    carla.command.SetAutopilot,
    carla.command.SetEnableGravity,
    carla.command.SetSimulatePhysics,
    carla.command.SetVehicleLightState,
    carla.command.SpawnActor,
]
