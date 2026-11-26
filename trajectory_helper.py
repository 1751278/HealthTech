#################
# trajectory_helper.py
# Created by Ethan Chan Nov 25 2026
# Last Updated: Nov 25 2026
# Description: Simple helper functions for trajectory processing in order to conserve space on nav
# TODO:
# - Add functions here when needed
#################
import math

def find_best_angle(trajectory, current_angle, prev_angle = None, angular_step=5, lookahead=100, distance_step = 10, smoothing = 0.01, bias = 10): #Pass in the list of (x, z) coordinates representing the trajectory. Lookahead is the max distance to consider
    x, z = trajectory[-1]
    trajectory = trajectory[-1000:]  # Only consider the last 1000 points for efficiency
    current_angle = int(current_angle - 90)# Adjust for coordinate system difference

    best_angle = None
    best_score = -float("inf")

    # Test every 10 degrees
    for angle in range(current_angle - 80, current_angle + 80, angular_step):
        rad = math.radians(angle)

        score = 0
        if prev_angle is not None:
            score += bias if angle == prev_angle else 0 #Add a bias to the score if the current angle matches the previous angle

        # Evaluate multiple distances ahead
        for distance in range(0, lookahead, distance_step):
            px = x + math.cos(rad) * distance
            pz = z + math.sin(rad) * distance

            # Distance to nearest recorded point
            nearest = min(
                math.hypot(px - vx, pz - vz)
                for vx, vz in trajectory[::3] #compare with only every 3 points
            )

            score += nearest

        if score > best_score * (1+smoothing): #Smoothing means the new score must be (smoothing*100)% more than the previous best score
            best_score = score
            best_angle = angle
            #print("BEST_SCORE: ", best_score)

    return best_angle