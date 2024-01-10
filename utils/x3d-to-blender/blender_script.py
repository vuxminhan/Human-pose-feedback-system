import bpy
import json
# Change to correct path
bpy.ops.import_anim.bvh(filepath="trainer-5.bvh")
bpy.ops.import_anim.bvh(filepath="learner-5.bvh")
bpy.ops.import_anim.bvh(filepath="learner-5.bvh")

with open("name_error_angles.json", 'r') as f:
    red_joint = json.load(f)
    
if bpy.context.object.type == 'ARMATURE':
    armature = bpy.data.objects["learner.001"]
    
    action = armature.animation_data.action if armature.animation_data else None
    
    if action:
        # Get the start and end frames of the animation
        start_frame = int(action.frame_range[0])
        end_frame = int(action.frame_range[1])
        print(start_frame,end_frame)
        # Iterate over the frames
        for frame in range(start_frame,end_frame-1):
        # Set the current frame
            bpy.context.scene.frame_set(frame)
            for bone in armature.data.bones:
                bone.select = True    
                bpy.context.object.data.bones[bone.name].color.palette = 'THEME01' 
                if bone.name in red_joint[frame]: 
                    bone.hide = False
                    bone.keyframe_insert("hide", frame=frame)
                else:
                    bone.hide = True
                    bone.keyframe_insert("hide", frame=frame)
                bone.select = False
    bpy.ops.object.posemode_toggle()

else: 
    print("Please select an armature object.")
