import bpy
import json
import os

# Remove all objects in the current scene
bpy.ops.object.select_all(action='SELECT')  # Select all objects
bpy.ops.object.delete()  # Delete selected objects

with open("ProvNERF Pose Generation\\blender\\camera_attributes.json", "r") as f:
    camera_data = json.load(f)

# Create a new camera
bpy.ops.object.camera_add(
    location=camera_data["location"], rotation=camera_data["rotation_euler"]
)
camera = bpy.context.object
camera.name = camera_data["name"]

# Set camera properties
camera.data.lens = camera_data["lens"]
camera.data.sensor_width = camera_data["sensor_width"]
camera.data.sensor_height = camera_data["sensor_height"]
camera.data.dof.use_dof = camera_data["dof_use_dof"]
camera.data.dof.aperture_fstop = camera_data["dof_aperture_fstop"]

bpy.context.scene.camera = camera

# Replace 'your_model_path.obj' with the actual path to your model file
# model_path = '/home/kla280/projects/def-keli/kla280/Human_Pose_Generation/hello_smpl.obj'


# Load from folder
# Set the path to the folder containing the .obj files
folder_path = "ProvNERF Pose Generation//3D_intermediate"

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".obj"):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Import the .obj file
        bpy.ops.import_scene.obj(filepath=file_path)
        print(f"Imported: {file_path}")


# Set up rendering settings (optional)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'  # Use the Cycles renderer
scene.render.filepath = "C:\\Users\\leois\\Downloads\\983-project\\Human_Pose_Generation\\ProvNERF Pose Generation\\blender\\render.png"  # Output path for rendered image

# Render the scene
bpy.ops.render.render(write_still=True)
