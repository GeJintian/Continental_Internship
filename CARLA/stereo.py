import glob
import os
import sys
import queue
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np


def generate_images():
    # Basic settings
    actor_list = []
    client = carla.Client('localhost',2000)
    client.set_timeout(2.0)
    world = client.get_world()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        #Create vehicles
        blueprint_library = world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle'))
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color',color)
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)

        focal_len = 1000
        fov = 360*np.arctan(1280/(2*focal_len))/np.pi
        baseline = 1
        # For depth camera
        camera_depth = blueprint_library.find('sensor.camera.depth')
        camera_depth.set_attribute("image_size_x",str(1280))
        camera_depth.set_attribute("image_size_y",str(720))
        camera_depth.set_attribute("fov",str(fov))
        camera_transform = carla.Transform(carla.Location(x=2,y=-0.5,z=1.4))
        camera_d = world.spawn_actor(camera_depth, camera_transform, attach_to=vehicle)
        actor_list.append(camera_d)
        cc = carla.ColorConverter.LogarithmicDepth
        depth_queue = queue.Queue()
        camera_d.listen(depth_queue.put)


        # For RGB camera
        # left camera
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(1280))
        cam_bp.set_attribute("image_size_y",str(720))
        cam_bp.set_attribute("fov",str(fov))
        camera_l = world.spawn_actor(cam_bp, camera_transform,attach_to=vehicle)
        actor_list.append(camera_l)
        l_queue = queue.Queue()
        camera_l.listen(l_queue.put)  
        # right camera
        camera_r = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2,y=0.5,z=1.4)), attach_to=vehicle)
        actor_list.append(camera_r)
        r_queue = queue.Queue()
        camera_r.listen(r_queue.put)

        # print out camera parameters
        print("image width",cam_bp.get_attribute("image_size_x").as_int())
        wide = cam_bp.get_attribute("image_size_x").as_int()
        print("image height",cam_bp.get_attribute("image_size_y").as_int())
        print("fov",cam_bp.get_attribute("fov").as_float())
        fov = cam_bp.get_attribute("fov").as_float()
        print("focal",wide / (2.0 * np.tan(fov * np.pi / 360.0)))

        for _ in range(0, 80):
            transform.location.x += 3.0

            bp = random.choice(blueprint_library.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('created %s' % npc.type_id)
        for _ in range(20):
            pedestrian_bp = random.choice(blueprint_library.filter('*walker.pedestrian*'))
            transform = random.choice(world.get_map().get_spawn_points())
            npc = world.try_spawn_actor(pedestrian_bp,transform)
            if npc is not None:
                actor_list.append(npc)
                print('people %s' % npc.type_id)
        time.sleep(1)
        counting_frame = 1
        start_frame = 5
        while True:
            world.tick()
            #print(counting_frame)
            image_d = depth_queue.get()
            image_l = l_queue.get()
            image_r = r_queue.get()            
            if counting_frame % start_frame == 0:#When the car is stable
                print(counting_frame)
                image_d.save_to_disk('D:\\Weeks\\CREStereo-master\\CREStereo-master\\img\\Carlap\\depth\\'+str(int(counting_frame/start_frame))+'.png')
                image_l.save_to_disk('D:\\Weeks\\CREStereo-master\\CREStereo-master\\img\\Carlap\\im0\\'+str(int(counting_frame/start_frame))+'.png')
                image_r.save_to_disk('D:\\Weeks\\CREStereo-master\\CREStereo-master\\img\\Carlap\\im1\\'+str(int(counting_frame/start_frame))+'.png')
            # fp = open("stereo_out/%06d.txt"%image_d.frame,'w')    
            # for i in actor_list:
            
            #     fp.write(str(i.bounding_box.get_world_vertices(i.get_transform())))
            # fp.close()
            counting_frame = counting_frame + 1
            #time.sleep(1)
    finally:
        camera_d.destroy()
        camera_l.destroy()
        camera_r.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

if __name__ == "__main__":
    generate_images()
