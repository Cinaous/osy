import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_datasets(PATH):
    '''
    加载数据集
    :return:
    '''
    train_monet = tf.data.Dataset.list_files(PATH + r'\trainA\*.jpg')
    train_photo = tf.data.Dataset.list_files(PATH + r'\trainB\*.jpg')
    test_monet = tf.data.Dataset.list_files(PATH + r'\testA\*.jpg')
    test_photo = tf.data.Dataset.list_files(PATH + r'\testB\*.jpg')

    return (train_monet, train_photo), (test_monet, test_photo)


def load(image_file):
    '''
    将图片加载成Tensorflow需要的格式
    :param image_file:
    :return:
    '''
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)

    return image


def show_img(img_path):
    '''
    传入图片路径展示图片
    :param img_path:
    :return:
    '''
    img = load(img_path)
    # casting to int for matplotlib to show the image
    plt.figure()
    plt.imshow(img / 255.0)
    plt.show()


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def resize(image, height, width):
    '''
    将图像调整为更大的高度和宽度
    :param image:
    :param height:
    :param width:
    :return:
    '''
    image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def random_crop(image, height, width):
    '''
    随机裁剪到目标尺寸
    :param image:
    :param height:
    :param width:
    :return:
    '''
    # 目标尺寸
    # IMG_WIDTH = 256
    # IMG_HEIGHT = 256
    cropped_image = tf.image.random_crop(
        image, size=[height, width, 3])

    return cropped_image


def random_jitter(image):
    '''
    将图像调整为更大的高度和宽度
    随机裁剪到目标尺寸
    随机将图像做水平镜像处理
    :param image:
    :return:
    '''
    # 调整大小为 286 x 286 x 3
    image = resize(image, 286, 286)

    # 随机裁剪到 256 x 256 x 3
    image = random_crop(image, 256, 256)

    # 随机镜像
    image = tf.image.random_flip_left_right(image)

    return image


# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def normalize(image):
    '''
    将图像归一化到区间 [-1, 1] 内
    :param image:
    :return:
    '''
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def preprocess_image_train(image_file):
    '''
    处理训练集图片
    :param image_file:
    :return:
    '''
    image = load(image_file)
    # image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image_file):
    '''
    处理测试集图片
    :param image_file:
    :return:
    '''
    image = load(image_file)
    image = normalize(image)
    return image


# def split_datasets(dataA, dataB):
#     '''
#     数据集切片操作，放入一个dataset中
#     :return:
#     '''
#     BUFFER_SIZE = 1000
#     BATCH_SIZE = 1
#
#     dataA = dataA.map(
#         preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
#         BUFFER_SIZE).batch(BATCH_SIZE)
#
#     dataB = dataB.map(
#         preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
#         BUFFER_SIZE).batch(BATCH_SIZE)
#
#     return dataA, dataB


# def next_data(data):
#     '''
#     建立迭代器，使每次取出1张图片
#     :return:
#     '''
#     return next(iter(data))

def calc_cycle_loss(real_image, cycled_image):
    '''
    定义循环一致损失函数

    图片X通过生成器G传递，该生成器生成图片Y^。
    生成的图片Y^通过生成器F传递，循环生成图片X^。
    在X和X^之间计算平均绝对误差。
    :param real_image:
    :param cycled_image:
    :return:
    '''
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def generate_images(model, test_input):
    '''
    定义图像生成函数
    :param model:
    :param test_input:
    :return:
    '''
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # 获取范围在 [0, 1] 之间的像素值以绘制它。
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.pause(5)
    plt.close()



def train_step(real_x, real_y):
    '''
    定义训练一次的函数
    :param real_x:
    :param real_y:
    :return:
    '''
    # persistent 设置为 Ture，因为 GradientTape 被多次应用于计算梯度。
    with tf.GradientTape(persistent=True) as tape:
        # 生成器 G 转换 X -> Y。
        # 生成器 F 转换 Y -> X。

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x 和 same_y 用于一致性损失。
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # 计算损失。
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # 总生成器损失 = 对抗性损失 + 循环损失。
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # 计算生成器和判别器损失。
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # 将梯度应用于优化器。
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))


if __name__ == '__main__':
    PATH = r'../horse2zebra'
    # show_img(PATH + r'\testA\00010.jpg')
    # img = preprocess_image_train(PATH + r'\testA\00010.jpg')
    # plt.imshow(img)
    # plt.show()

    # 读取数据集 --------------------------------------------------------------
    (train_monet, train_photo), (test_monet, test_photo) = get_datasets(PATH)
    # -----------------------------------------------------------------------

    # 将训练集所有图片进行切片操作，放入一个dataset中 ------------------------------
    BUFFER_SIZE = 1000
    BATCH_SIZE = 1

    train_horses = train_monet.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_zebras = train_photo.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
    # ------------------------------------------------------------------------

    # 将测试集所有图片进行切片操作，放入一个dataset中 -----------------------------
    test_horses = test_monet.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_zebras = test_photo.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
    # ----------------------------------------------------------------------

    # 建立迭代器，使每次取出1张图片 ------------
    sample_horse = next(iter(train_horses))
    # sample_zebra = next(iter(train_zebras))

    # 导入 Pix2Pix 模型 ------------------------------------------------------------
    OUTPUT_CHANNELS = 3

    generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
    # -----------------------------------------------------------------------------

    # 加载已训练训练参数 ---------------------------------------------
    generator_g__wt_pth = r'./weights/generator_g.ckpt'
    generator_f__wt_pth = r'./weights/generator_f.ckpt'
    discriminator_x__wt_pth = r'./weights/discriminator_x.ckpt'
    discriminator_y__wt_pth = r'./weights/discriminator_y.ckpt'
    if os.path.exists(generator_g__wt_pth + '.index'):
        print('加载生成模型g')
        generator_g.load_weights(generator_g__wt_pth)
    if os.path.exists(generator_f__wt_pth + '.index'):
        print('加载生成模型f')
        generator_f.load_weights(generator_f__wt_pth)
    if os.path.exists(discriminator_x__wt_pth + '.index'):
        print('加载判别模型x')
        discriminator_x.load_weights(discriminator_x__wt_pth)
    if os.path.exists(discriminator_y__wt_pth + '.index'):
        print('加载判别模型y')
        discriminator_y.load_weights(discriminator_y__wt_pth)
    # ----------------------------------------------------------

    # 定义判别器损失函数 ------------------------------------------------
    LAMBDA = 10
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def discriminator_loss(real, generated):
        real_loss = loss_obj(tf.ones_like(real), real)

        generated_loss = loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5


    # -----------------------------------------------------------------

    # 定义生成器损失函数 -------------------------------------
    def generator_loss(generated):
        return loss_obj(tf.ones_like(generated), generated)


    # -----------------------------------------------------

    # 定义一致性损失函数 ----------------------------------------
    def identity_loss(real_image, same_image):
        '''

        :param real_image:
        :param same_image:
        :return:
        '''
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss


    # --------------------------------------------------------

    # 初始化优化器 ---------------------------------------------------------
    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    # --------------------------------------------------------------------

    # 训练 ----------------------
    EPOCHS = 20
    for epoch in range(EPOCHS):
        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
            train_step(image_x, image_y)
            # print('.', end='')
            print('当前第 {0} 轮， 第 {1} 次训练'.format(epoch, n))

            # 保存模型 ----------------------------------------------
            if n % 50 == 0:
                print('保存模型中...')
                generator_g.save_weights(generator_g__wt_pth)
                generator_f.save_weights(generator_f__wt_pth)
                discriminator_x.save_weights(discriminator_x__wt_pth)
                discriminator_y.save_weights(discriminator_y__wt_pth)
                print('保存完毕')
                # -------------------------------------------------------
                # 使用一致的图像（sample_horse），以便模型的进度清晰可见。
                generate_images(generator_g, sample_horse)

            n += 1

        clear_output(wait=True)
        # 使用一致的图像（sample_horse），以便模型的进度清晰可见。
        generate_images(generator_g, sample_horse)
    # -----------------------------

    # 测试 -------------------------------
    # 在测试数据集上运行训练的模型。
    for inp in test_horses.take(5):
        generate_images(generator_g, inp)
    # -----------------------------------
