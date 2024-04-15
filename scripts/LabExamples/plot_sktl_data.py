import matplotlib.pyplot as plt

# # Given skeletal data
# skeletal_data_std = {
# "body_parts": {
#         "Neck": {"x": 0.41310975609756095, "y": 0.5},
#         "RShoulder": {"x": 0.36585365853658536, "y": 0.5407608695652174},
#         "RElbow": {"x": 0.35823170731707316, "y": 0.5788043478260869},
#         "RWrist": {"x": 0.3521341463414634, "y": 0.5788043478260869},
#         "RHip": {"x": 0.4115853658536585, "y": 0.5679347826086957}, 
#         "RKnee": {"x": 0.41615853658536583, "y": 0.6494565217391305}
#         } 
# }
# skeletal_data_sit = {
# "body_parts": {
#         "Nose": {"x": 0.6432926829268293, "y": 0.4673913043478261}, 
#         "Neck": {"x": 0.760670731707317, "y": 0.532608695652174},
#         "RShoulder": {"x": 0.8307926829268293, "y": 0.532608695652174},
#         "LShoulder": {"x": 0.7149390243902439, "y": 0.529891304347826},
#         "RHip": {"x": 0.6615853658536586, "y": 0.720108695652174},
#         "RKnee": {"x": 0.5884146341463414, "y": 0.6766304347826086}, 
#         "RAnkle": {"x": 0.5533536585365854, "y": 0.6820652173913043},
#         "LHip": {"x": 0.7240853658536586, "y": 0.7065217391304348},
#         "REye": {"x": 0.6463414634146342, "y": 0.45108695652173914},
#         "LEye": {"x": 0.6585365853658537, "y": 0.453804347826087},
#         "LEar": {"x": 0.7317073170731707, "y": 0.46467391304347827}
#         } 
# }

# Given skeletal data
skeletal_data_std = {
"body_parts": {
        "Neck": {"x": 0.41310975609756095, "y": 0.5},
        "RShoulder": {"x": 0.36585365853658536, "y": 0.5407608695652174},
        "RKnee": {"x": 0.41615853658536583, "y": 0.6494565217391305}
        } 
}
skeletal_data_sit = {
"body_parts": {
        "Neck": {"x": 0.760670731707317, "y": 0.532608695652174},
        "RShoulder": {"x": 0.8307926829268293, "y": 0.532608695652174},
        "RKnee": {"x": 0.5884146341463414, "y": 0.6766304347826086}, 
        } 
}

##JUST NEED TO CHEKC UPPER AND LOWER BODY 

##Anything below the hips is lower body, if x of upperbody is diff than lower then sitting .

# Define connections between body parts
skeleton = [
    ("Neck", "RShoulder"),
    ("RShoulder", "RKnee"),
]

skeletal_data = skeletal_data_sit

# Plotting
plt.figure(figsize=(6, 8))

# Plot points
for part, info in skeletal_data['body_parts'].items():
    plt.plot(info['x'], info['y'], 'o', label=part)  # Plot each body part as a point

# Draw lines for the skeleton
for connection in skeleton:
    start_point = skeletal_data['body_parts'][connection[0]]
    end_point = skeletal_data['body_parts'][connection[1]]
    plt.plot([start_point['x'], end_point['x']], [start_point['y'], end_point['y']], 'k-')  # 'k-' denotes a black line

plt.gca().invert_yaxis()  # Inverting y-axis for better visualization
plt.title('Skeletal Structure')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
