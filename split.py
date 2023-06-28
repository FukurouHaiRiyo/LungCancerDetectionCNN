import splitfolders
splitfolders.ratio('/home/andrei/Desktop/ProiectLicenta/lung_image_sets', 
      output="output", 
      seed=1337, 
      ratio=(.8, 0.1,0.1)
)