import math


def horizontal_plus_offset_joint4_deg(joint2_deg, joint3_deg, align_wrist_pitch_deg):
    return -joint2_deg - joint3_deg + align_wrist_pitch_deg


def joint4_rad_from_joint23(joint2_rad, joint3_rad, align_wrist_pitch_deg):
    return math.radians(
        horizontal_plus_offset_joint4_deg(
            math.degrees(joint2_rad),
            math.degrees(joint3_rad),
            align_wrist_pitch_deg,
        )
    )


def pose_from_q1234(q1234, joint6_rad):
    return [
        float(q1234[0]),
        float(q1234[1]),
        float(q1234[2]),
        float(q1234[3]),
        float(joint6_rad),
    ]

