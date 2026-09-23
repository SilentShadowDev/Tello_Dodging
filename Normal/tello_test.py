from djitellopy import Tello

tello = Tello()
tello.connect()

print(f"Battery: {tello.get_battery()}%")
