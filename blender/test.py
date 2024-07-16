import bpy

# Replace 'your_model_path.obj' with the actual path to your model file
model_path = '/home/kla280/projects/def-keli/kla280/Human_Pose_Generation/hello_smpl.obj'

# Load the model
bpy.ops.import_scene.obj(filepath=model_path)

# Optionally, manipulate the scene or model
# Example: Rotate the model
bpy.context.object.rotation_euler[2] = 0.5  # Rotate by 0.5 radians around the Z axis

# Set up rendering settings (optional)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'  # Use the Cycles renderer
scene.render.filepath = 'render.png'  # Output path for rendered image

# Render the scene
bpy.ops.render.render(write_still=True)
