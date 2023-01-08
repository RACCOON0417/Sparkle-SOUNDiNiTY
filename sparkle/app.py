from flask import Flask,render_template
from pyngrok import conf, ngrok

from scipy.io.wavfile import write

import os

# 모델 로딩
import tensorflow.compat.v1 as tf
import numpy as np

def ganSound(model):
    ckpt={
        'Wood':'1911',
        'Train':'12131',
        'Snow':'0',
        'Reload':'4463',
        'Gun_One':'34387'
    }
    tf.disable_v2_behavior()
    # Load the graph
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('./static/Model/Model_Checkpoint_'+model+'/infer/infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, './static/Model/Model_Checkpoint_'+model+'/model.ckpt-'+ckpt[model])

    # Create 50 random latent vectors z
    _z = (np.random.rand(50, 100) * 2.) - 1
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(G_z, {z: _z})

    if os.path.exists('./static/sound/'+model+'_temp.wav'):
        os.remove('./static/sound/'+model+'_temp.wav')
        print("remove file")
    # with open('./static/sound/'+model+'_temp.wav','wb') as myfile:
    #     myfile.write(Audio(_G_z[0, :, 0], rate=16000).data)

    write('./static/sound/'+model+'_temp.wav', 16000, _G_z[0, :, 0])


app = Flask(__name__)

# Ngrok
conf.get_default().region = "jp"
conf.get_default().auth_token = "2IG1beK1gUJLxHkm3e6nTMHdDyA_3GRvJr32zL6528qLAiurc"
http_tunnel = ngrok.connect(7777)

tunnels = ngrok.get_tunnels()
for kk in tunnels:
    print(kk.public_url)
    #print(ngrok.disconnect(kk.public_url))

# 메인페이지
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/player/<model>', methods=['GET'])
def player(model):
    ganSound(model)
    return render_template('player.html', model=model)

@app.route('/info', methods=['GET'])
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True, use_reloader=False)



# 음원 생성
# import tensorflow.compat.v1 as tf
# from IPython.display import display, Audio
# import numpy as np
# from scipy.io.wavfile import read
# tf.disable_v2_behavior()
# # Load the graph
# tf.reset_default_graph()
# saver = tf.train.import_meta_graph('/content/drive/MyDrive/DNA_HERO/Hyung/Model_Checkpoint_Reload/infer/infer.meta')
# graph = tf.get_default_graph()
# sess = tf.InteractiveSession()
# saver.restore(sess, '/content/drive/MyDrive/DNA_HERO/Hyung/Model_Checkpoint_Reload/model.ckpt-4452')

# # Create 50 random latent vectors z
# _z = (np.random.rand(50, 100) * 2.) - 1
# z = graph.get_tensor_by_name('z:0')
# G_z = graph.get_tensor_by_name('G_z:0')
# _G_z = sess.run(G_z, {z: _z})

# # Play audio in notebook

# #display(Audio(_G_z[0, :, 0], rate=16000))
# with open('./static/sound/test.wav', 'wb') as f:
#    f.write(Audio(_G_z[0, :, 0], rate=16000).data)