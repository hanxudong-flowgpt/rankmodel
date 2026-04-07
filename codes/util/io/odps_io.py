import tensorflow as tf
import time
from util.tflog import tflogger as logging


class TableWriter():
  def __init__(self,name):
    self.writer = open(name,'w')

  def write(self,blockid,records):
    for x in records:
      self.writer.write('%s:\t%s\n'%(str(blockid),'\t\t'.join(x)))

  def close(self):
    self.writer.close()


def resetOdpsTable(outputtable, task_id, local_mode, odps_user=""):
  if not local_mode:
    import odps as Odps
    from odps.models import Schema
    if task_id == 0:
      # odps.delete_table(outputtable, "search_offline_dev", if_exists=True)
      schema = Schema.from_lists(['id', 'predict', 'label'],
                                 ['STRING', 'STRING', 'STRING'], ['ds'], ['string'])
      table = odpsoutput.create_table(outputtable, schema, if_not_exists=True)
    return odpsoutput
  else:
    return None


def getTableWriter(odpsinput, outputtable, task_id, ds_output, local_mode):
  if not local_mode:
    while True:
      if odpsinput.exist_table(outputtable):
        table = odpsinput.get_table(outputtable)
        logging.info("Get table %s Done"%outputtable)
        break
      else:
        logging.info("Waiting table %s to be created..."%outputtable)
        time.sleep(60)

    logging.info('wait for table init done:%s/ds=%s, task_id=%s' % (outputtable, ds_output, str(task_id)))
    time.sleep(10)
    logging.info('wait done')
    tablew = table.open_writer(partition="ds=%s" % ds_output, blocks=[task_id], create_partition=True,
                               reopen=True)
  else:
    tablew = TableWriter('fake_data/%s_ds=%s.txt' % (outputtable, ds_output))
  return tablew