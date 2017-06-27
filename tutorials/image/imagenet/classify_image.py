# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import threading
import SimpleHTTPServer

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

gQueueReady = threading.Event()

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    return list(map(lambda node_id: (node_lookup.id_to_string(node_id), predictions[node_id]), top_k))

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def write_file(name, title="Waiting for an image", delay=10):
  s = ("<html><head><title>{0}</title>"
      '<meta http-equiv="refresh" content="{1}" >'
      '</head>'
      '<body><center><img src="img.jpg"><br><h2>{0}</h2></center></body>'
      '</html>').format(title, delay)
  with open("index.html", 'w') as file:
      file.write(s)

class MyHTTPHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
  def do_GET(self):
    SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
    if self.path == "/img.jpg":
      gQueueReady.set()
      print("ready!", self.path)

def serve_http():
  import SocketServer
  PORT = FLAGS.listen_port
  write_file("index.html", "Waiting for first image")
  Handler = MyHTTPHandler
  httpd = SocketServer.TCPServer(("", PORT), Handler)
  print("serving at http://0.0.0.0:%d" % (PORT))
  httpd.serve_forever()

def pull_from_minio():
  import tempfile
  tmpdir = tempfile.mkdtemp(suffix="image-ai")
  import os
  import time

  os.chdir(tmpdir)
  t = threading.Thread(target=serve_http)
  t.daemon = True
  t.start()

  while True:
    import urllib2
    import json
    # resp = '{"job":{"id":"8f8d8390-5545-11e7-8950-83b1c5d7f8d9","data":{"Key":"fook/rds.xlsx","msg":"","time":"2017-06-19T23:18:10Z","level":"info","Records":[{"s3":{"bucket":{"arn":"arn:aws:s3:::fook","name":"fook","ownerIdentity":{"principalId":"AKIAIOSFODNN7EXAMPLE"}},"object":{"key":"rds.xlsx","eTag":"12d8794b80a50209f3ed60adaa1aecaa","size":50166,"sequencer":"14C9A91C74A2620C","versionId":"1","contentType":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet","userDefined":{"content-type":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}},"configurationId":"Config","s3SchemaVersion":"1.0"},"source":{"host":"","port":"","userAgent":""},"awsRegion":"us-east-1","eventName":"s3:ObjectCreated:Put","eventTime":"2017-06-19T23:18:10Z","eventSource":"minio:s3","eventVersion":"2.0","userIdentity":{"principalId":"AKIAIOSFODNN7EXAMPLE"},"responseElements":{"x-amz-request-id":"14C9A91C74A2620C","x-minio-origin-endpoint":"http://10.244.0.62:9000"},"requestParameters":{"sourceIPAddress":"10.244.0.1:49341"}}],"EventType":"s3:ObjectCreated:Put"},"name":"webhook"}}'
    response = urllib2.urlopen(FLAGS.queue_fetch)
    resp = response.read()
    print(resp)
    obj = json.loads(resp)
    job = obj['job']
    dest_file = None
    if job:
      data = job['data']
      # print(data.keys())
      # print(data['Key'])
      for record in data['Records']:
        eventName = record['eventName']
        if record['eventName'] == 's3:ObjectCreated:Put':
          key = record['s3']['object']['key']
          bucket = record['s3']['bucket']['name']
          # print(bucket, key)
          import boto3
          # boto3.set_stream_logger(name='botocore')
          from botocore.client import Config
          s3 = boto3.resource('s3', endpoint_url=FLAGS.s3_url,
                              config=Config(signature_version='s3v4'),
                              aws_access_key_id=FLAGS.s3_access_key,
                              aws_secret_access_key=FLAGS.s3_secret_key,
                              region_name='us-east-1')
          try:
            dest_file = "tmp.img"
            s3.Bucket(bucket).download_file(key, dest_file)
          except Exception as e:
            print ("Failed to download %s from minio, probly another filename encoding issue" % key, e)
            dest_file = None
          if dest_file:
            ret = run_inference_on_image(dest_file)
            if len(ret):
              print(ret)
              os.rename(dest_file, "img.jpg")
              write_file("index.html", str(ret[0]))
        else:
          print(eventName)
        import urllib2
        req = urllib2.Request(FLAGS.queue_complete, job['id'])
        response = urllib2.urlopen(req)
        response_txt = response.read()
        print("complete", job['id'], response_txt)
        if dest_file:
          # wait for http serve to serve this image
          print("waiting to serve...")
          gQueueReady.wait()
    else:
      time.sleep(1)

def main(_):
  maybe_download_and_extract()
  if FLAGS.queue_fetch != '':
    return pull_from_minio()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  print(run_inference_on_image(image))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  parser.add_argument(
      '--queue_fetch',
      type=str,
      default='',
      help='Event queue url to get jobs to images from s3'
  )
  parser.add_argument(
      '--queue_complete',
      type=str,
      default='',
      help='Event queue url to complete jobs'
  )
  parser.add_argument(
      '--s3_url',
      type=str,
      default='',
      help='S3 URL'
  )
  parser.add_argument(
      '--s3_access_key',
      type=str,
      default='',
      help='S3 access key'
  )
  parser.add_argument(
      '--s3_secret_key',
      type=str,
      default='',
      help='S3 secret access key'
  )
  parser.add_argument(
      '--listen_port',
      type=int,
      default=8000,
      help='Listen port'
  )

  FLAGS, unparsed = parser.parse_known_args()
  try:
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  except Exception as e:
    print("Fatal exception, exiting", e)
    sys.exit(1)
