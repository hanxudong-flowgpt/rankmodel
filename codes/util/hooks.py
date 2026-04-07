import tensorflow as tf
import logging
from tensorflow.python.saved_model.utils_impl import get_variables_path
import os
import time


class ExportModelHook(tf.train.SessionRunHook):
    def __init__(self, export_root, do_export, done_list, tracker_list, input_dict, output_dict, is_local=False, tracker=''):
        self.export_root = export_root
        self.do_export = do_export
        self.done_list = done_list
        self.tracker_list = tracker_list
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.is_local = is_local
        self.tracker = tracker

    def end(self, session):
        pass

    def export_sm(self, session, ts=None):
        timestamp = int(time.time()) if ts is None else int(ts)
        if self.do_export:
            session.graph._unsafe_unfinalize()
            export_dir = os.path.join(self.export_root, str(timestamp))
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            signature = tf.saved_model.predict_signature_def(inputs=self.input_dict, outputs=self.output_dict)

            builder.add_meta_graph_and_variables(sess=session,
                                                 tags=[tf.saved_model.tag_constants.SERVING],
                                                 clear_devices=True,
                                                 signature_def_map={
                                                     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
            builder.save()
            session.graph.finalize()
            if self.is_local:
                # self._update_done_list_local(str(timestamp))
                self._update_tracker_list_local(str(timestamp), self.tracker)
            else:
                # self._update_done_list_hadoop(str(timestamp))
                self._update_tracker_list_hadoop(str(timestamp), self.tracker)
        return timestamp

    def write_donefile(self, timestamp):
        if self.is_local:
            self._update_done_list_local(str(timestamp))
        else:
            self._update_done_list_hadoop(str(timestamp))

    def _update_done_list_local(self, timestamp):
        if not os.path.exists(self.done_list):
            logging.info('done_list dose not exist before')

        with open(self.done_list, 'a') as f:
            f.write(timestamp + '\n')
            logging.info('update local done list with timestamp: {}'.format(timestamp))

    def _update_tracker_list_local(self, timestamp, tracker):

        if not os.path.exists(self.tracker_list):
            logging.info('tracker_list does not exist before')

        with open(self.tracker_list, 'a') as f:
            f.write(timestamp + ';' + tracker + '\n')
            logging.info('update local tracker list with tracker: {}'.format(tracker))

    def _update_done_list_hadoop(self, timestamp):
        is_file_in_hadoop = os.system("hadoop fs -get %s" % self.done_list)
        done_list_file_name = os.path.basename(self.done_list)
        if is_file_in_hadoop == 0:
            with open(done_list_file_name, 'a') as f:
                f.write(timestamp + '\n')
            rm_status = os.system("hadoop fs -rm %s" % self.done_list)
            if rm_status == 0:
                logging.info('remove original done_list file')
            else:
                logging.info('remove original done_list failed')
        else:
            with open(done_list_file_name, 'a') as f:
                f.write(timestamp + '\n')
        logging.info('update hadoop done list with timestamp: {}'.format(timestamp))
        write_args = "%s %s" % (done_list_file_name, os.path.dirname(self.done_list))
        write_status = os.system("hadoop fs -put -f %s" % write_args)
        if write_status == 0:
            logging.info("###########LOG INFO Sucessfully upload file %s" % self.done_list)
        else:
            logging.info("###########ERROR INFO Failed to upload file %s" % self.done_list)

    def _update_tracker_list_hadoop(self, timestamp, tracker):
        is_file_in_hadoop = os.system("hadoop fs -get %s" % self.tracker_list)
        tracker_list_file_name = os.path.basename(self.tracker_list)
        if is_file_in_hadoop == 0:
            with open(tracker_list_file_name, 'a') as f:
                f.write(timestamp + ';' + tracker + '\n')
            rm_status = os.system("hadoop fs -rm %s" % self.tracker_list)
            if rm_status == 0:
                logging.info('remove original tracker_list file')
            else:
                logging.info('remove original tracker_list failed')
        else:
            with open(tracker_list_file_name, 'a') as f:
                f.write(timestamp + ';' + tracker + '\n')
        logging.info('update local tracker list with tracker: {}'.format(tracker))
        write_args = "%s %s" % (tracker_list_file_name, os.path.dirname(self.tracker_list))
        write_status = os.system("hadoop fs -put -f %s" % write_args)
        if write_status == 0:
            logging.info("###########LOG INFO Sucessfully upload file %s" % self.tracker_list)
        else:
            logging.info("###########ERROR INFO Failed to upload file %s" % self.tracker_list)


class LoadModelHook(tf.train.SessionRunHook):
    def __init__(self, export_root, do_load, done_list, ckpt_dir, model_timestamp='', is_local=False):
        self.export_root = export_root
        self.do_load = do_load
        self.done_list = done_list
        self.ckpt_dir = ckpt_dir
        self.model_timestamp = model_timestamp
        self.is_local = is_local

    def begin(self):
        self.ckpt_cur = tf.train.latest_checkpoint(self.ckpt_dir)
        self.global_step_tensor = tf.train.get_or_create_global_step()
        self.reset_global_step = tf.assign(self.global_step_tensor, 0)
        self.last_timestamp = self.try_get_last_timestamp_from_local(self.done_list) if self.is_local \
            else self.try_get_last_timestamp_from_hadoop(self.done_list)
        logging.info('last timestamp is {}'.format(self.last_timestamp))
        if self.model_timestamp == '':
            self.load_export_dir = os.path.join(self.export_root,
                                            self.last_timestamp) if self.last_timestamp is not None else None
        else:
            self.load_export_dir = os.path.join(self.export_root,
                                            self.model_timestamp)
        self.saver = tf.train.Saver(sharded=True)

    def try_get_last_timestamp_from_hadoop(self, done_list):
        is_file_in_hadoop = os.system("hadoop fs -get %s" % done_list)
        if is_file_in_hadoop == 0:
            local_done_list = os.path.basename(done_list)
            timestamp = self.try_get_last_timestamp_from_local(local_done_list)
        else:
            logging.info('done_list file dosen\'t exist')
            timestamp = None
        return timestamp

    def try_get_last_timestamp_from_local(self, done_list):
        if not os.path.exists(done_list):
            logging.info('done_list file dose not exist')
            return None

        with open(done_list, 'r') as f:
            lines = f.readlines()
        timestamp = lines[-1].strip('\n') if len(lines) > 0 else None
        return timestamp

    def after_create_session(self, session, coord):
        if self.load_export_dir is not None and self.ckpt_cur is None:
            if not self.do_load:
                logging.info('do_load set to false, no parameter restored')
                return
            self.saver.restore(session, get_variables_path(self.load_export_dir))
            logging.info('loading model from {}'.format(self.load_export_dir))
            new_step = session.run(self.reset_global_step)
            logging.info('reset global step to {}'.format(new_step))
        else:
            if self.load_export_dir is not None:
                logging.info(
                    'expect to load model from {}, but current checkpoint dir {} found'.format(self.load_export_dir,
                                                                            self.ckpt_dir))
            else:
                logging.info('last model directory not found.')
