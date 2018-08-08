
# coding: utf-8

# <font color='purple'> The purpose of this algorithm is to take an image and play around with it. Then the new images can be taken and put through the Google Cloud Platform AutoML Beta for image classification </font>

# In[123]:


# Importing of libraries used 
import matplotlib
import sys
import random
import numpy as np
import PIL
 
# Convert Image to array
img = PIL.Image.open(r"G:\My Drive\CATSVDOGS2.0\test1\1.jpg")
a = numpy.array(img)
print(a)
 
# Convert array to Image
img = PIL.Image.fromarray(a)
img.show()


# In[144]:


# This will transpose the initial array, then shuffle the columns (now in row form thanks to the transpose).
# Then finally it converts it back to its original form with the shuffle performed

a=np.transpose(a)
np.random.shuffle(a)
a=np.transpose(a)
print(a)


# In[125]:


# Transform Matrix 
b = np.random.randint(1,3,size=(499,381,3))


# In[126]:


# Check to see if it matches with a
print(b)
print(b.shape)


# In[127]:


# Output matrix: empty array of same dimensions as a and b
c = np.zeros(shape=(499,381,3))


# In[128]:


# Checks if correct
print(c)
print(c.shape)


# In[129]:


# Convert array to Image
img = PIL.Image.fromarray(a)
img.show()


# <font color='purple'> So the code ends up being able to shuffle the matrix and spit back out an image. This only shuffles the RGB values however and does not move the pixels around.</font>

# <font color='purple'> Below I will take my first image in the CATDOG dataset and jumble it. It will then save into a new folder and show the result.. However this will not be consistent across a dataset as it jumbles the image differently every time</font>

# In[149]:


BLOCKLEN = 30 # Adjust and be careful here. The higher the value the less jumbled the image becomes... and vice-versa!
img = PIL.Image.open(r"G:\My Drive\CATSVDOGS2.0\test1\1.jpg")
width, height = img.size
xblock = width / BLOCKLEN
yblock = height / BLOCKLEN
blockmap = [(xb*BLOCKLEN, yb*BLOCKLEN, (xb+1)*BLOCKLEN, (yb+1)*BLOCKLEN) 
            for xb in range(int(xblock)) for yb in range(int(yblock))]
shuffle = list(blockmap)
random.shuffle(shuffle)
result = Image.new(img.mode, (width, height))
for box, sbox in zip(blockmap, shuffle):
    c = img.crop(sbox)
    result.paste(c, box)
#result.save(r"G:\My Drive\CATSVDOGS2.0\test1\00000000001.jpg")
result.show()


# <font color='purple'> The below sets the font size for the script (not part of jumble algorithm) </font>

# In[139]:


from IPython.core.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 20px; }</style>"))


# <font color='purple'> </font>
