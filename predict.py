import os
if os.path.exists(os.path.join("models", "fruit_resnet50_best.pt")):
    print('MODEL SAVED SUCCESSFULLY')
else:
    print('MODEL NOT FOUND')
