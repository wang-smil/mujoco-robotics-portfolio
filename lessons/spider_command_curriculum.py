"""小蜘蛛平地速度课程和各检查点共享的验收路线。"""


STOP = (0., 0., 0.)
FORWARD = tuple((speed, 0., 0.) for speed in (.018, .034, .046))
BACKWARD = tuple((-speed, 0., 0.) for speed in (.018, .034, .046))
LEFT = tuple((0., speed, 0.) for speed in (.004, .010, .014))
RIGHT = tuple((0., -speed, 0.) for speed in (.004, .010, .014))
TURN_LEFT, TURN_RIGHT = (0., 0., .25), (0., 0., -.25)
ARC_LEFT, ARC_RIGHT = (.034, 0., .18), (.034, 0., -.18)
FRONT_BACK = (FORWARD[1], BACKWARD[0], FORWARD[1], STOP)
BACK_FRONT = (BACKWARD[1], FORWARD[0], BACKWARD[1], STOP)


AXIAL = tuple((command,) * 4 for command in (STOP, *FORWARD, *BACKWARD)) + (
    (STOP, *FORWARD), (STOP, *BACKWARD), (FORWARD[2], FORWARD[0], FORWARD[1], STOP),
)
SIDE = tuple((command,) * 4 for command in (*LEFT, *RIGHT, TURN_LEFT, TURN_RIGHT)) + (
    (STOP, *LEFT), (STOP, *RIGHT), (TURN_LEFT, STOP, TURN_RIGHT, STOP),
)
TRANSITIONS = (
    (ARC_LEFT,) * 4, (ARC_RIGHT,) * 4, FRONT_BACK, BACK_FRONT,
    (FORWARD[1], ARC_LEFT, FORWARD[1], STOP), (FORWARD[1], ARC_RIGHT, FORWARD[1], STOP),
    (LEFT[1], RIGHT[1], LEFT[0], STOP), (TURN_RIGHT, TURN_LEFT, FORWARD[0], STOP),
)
CURRICULUM = {
    "axial": AXIAL,
    "strafe_turn": (*AXIAL, *SIDE, *SIDE),
    "mixed": (*AXIAL, *SIDE, *TRANSITIONS, *TRANSITIONS),
}
DEMONSTRATOR_PROFILES = (*AXIAL, *SIDE, *TRANSITIONS)


def validation_routes():
    routes = {"stop": [(3.6, STOP)]}
    for name, commands in (("forward", FORWARD), ("backward", BACKWARD), ("left", LEFT), ("right", RIGHT)):
        routes.update({f"{name}_{level}": [(3.6, command)]
                       for level, command in zip(("low", "medium", "high"), commands)})
    routes.update({name: [(3.6, command)] for name, command in (
        ("turn_left", TURN_LEFT), ("turn_right", TURN_RIGHT), ("arc_left", ARC_LEFT), ("arc_right", ARC_RIGHT),
    )})
    routes["forward_reverse"] = list(zip((2.6, 3.2, 2.8, 2.0), FRONT_BACK))
    routes["reverse_forward"] = list(zip((3.1, 2.7, 3.0, 2.0), BACK_FRONT))
    return routes
