import tensorflow as tf
import pathlib
import os
data_dir = pathlib.Path(os.path.abspath('webs/static/ImageToTrain'))

#Tạo tập dữ liệu
batch_size =3
img_height=224
img_width=224


#Chỉ dùng 80% hình ahr để đào tạo và 20% hình ảnh để xác thực
train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size
)
val_ds=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
class_names=train_ds.class_names