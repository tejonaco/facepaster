# FACEPASTER

Script to paste faces into other images, it can be used from facepaster.py  
In adition there is a Telegram bot implementation of the script  

## REQUERIMENTS  
Python (made in 3.8)  
facenet_pytorch  
numpy  
pillow  
matplotlib (optional, only to plot images)  
python-telegram-bot (only for the bot implementation)  
torch  
torchvision  
opencv-python  

## USAGE  
### as library:
    from facepaster import FacePaster  
    with FacePaster(input_face) as fp:  
        img = fp.paste_faces(input_img)  
        fp.plot(img) #optional  

### in command line:* 
    python facepaster.py [-h] [-o OUTPUT] [-p] face img

    
## TODO  
make a translation engine for the bot  
