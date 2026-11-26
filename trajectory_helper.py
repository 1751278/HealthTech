#################
# trajectory_helper.py
# Created by Ethan Chan Nov 25 2026
# Last Updated: Nov 25 2026
# Description: Simple helper functions for trajectory processing in order to conserve space on nav
# TODO:
# - Add functions here when needed
#################
import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon

def find_best_traj_angle(trajectory, current_angle, prev_angle = None, angular_step=5, lookahead=100, distance_step = 10, smoothing = 0.01, bias = 10): #Pass in the list of (x, z) coordinates representing the trajectory. Lookahead is the max distance to consider
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

def find_best_navigation_direction(frame, floor_seg, traj_angle, angular_step=10, min_ray_length=80, clearance_width=80):
    h, w = frame.shape[:2]
    polygon = Polygon(floor_seg)
    start_point = (w//2, h)  # (x, y) in cv2 coordinate system
    max_ray_length = math.hypot(h, w) + 1  # Should be larger than your image diagonal
    ray_lengths = [] # will contain (distance to edge of mask, angle_deg)
    for angle_deg in range (-80, 80, angular_step):
        ray_lengths.append(find_distance_edge(start_point, max_ray_length, angle_deg, polygon))

    maxima_values = [ray_lengths[i] for i in range(1, len(ray_lengths) - 1) 
                    if ray_lengths[i - 1][0] < ray_lengths[i][0] > ray_lengths[i + 1][0]]#finds the local maxima
    if ray_lengths[0][0]>ray_lengths[1][0]:# find if end points are maxima too
        maxima_values.append(ray_lengths[0])
    if ray_lengths[-1][0]>ray_lengths[-2][0]:
        maxima_values.append(ray_lengths[-1])

    filtered = [lengths for lengths in maxima_values if lengths[0]>min_ray_length]#filter out any rays that are not long enough

    enough_clearance = []#find if there is enough clearance to go through path
    for distance, angle_deg in filtered:
        if (find_distance_edge(start_point-(clearance_width//2), max_ray_length, angle_deg, polygon)[0] > distance*0.6 and
            find_distance_edge(start_point+(clearance_width//2), max_ray_length, angle_deg, polygon)[0] > distance*0.6):
            enough_clearance.append((distance, angle_deg))

    best_angle = None
    if enough_clearance:
        possible_angles = [data[1] for data in enough_clearance]
        best_angle = min(possible_angles, key=lambda x: abs(x - traj_angle))
    return best_angle


# find the distance from point to edge of seg from angle
def find_distance_edge(start_point, max_ray_length, angle_deg, polygon):
    angle_rad = math.radians(angle_deg)
    end_x = start_point[0] + max_ray_length * math.cos(angle_rad)
    end_y = start_point[1] + max_ray_length * math.sin(angle_rad)
    ray = LineString([start_point, (end_x, end_y)])

    # 5. Find intersection with the polygon's exterior boundary
    intersection = ray.intersection(polygon.exterior)

    if not intersection.is_empty:
        origin = Point(start_point)
        # Handle cases where the ray hits multiple edges (e.g., concave shapes)
        if intersection.geom_type == "Point":
            distance = origin.distance(intersection)
            return (distance, angle_deg)

        elif intersection.geom_type in ["MultiPoint", "GeometryCollection"]:
            # Find the closest intersection point to the origin
            points = [geom for geom in intersection.geoms if geom.geom_type == "Point"]
            closest = min(points, key=lambda p: origin.distance(p))
            distance = origin.distance(closest)
            return (distance, angle_deg)
    else:
        return (0, angle_deg)