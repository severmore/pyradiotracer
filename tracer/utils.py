class ProgressBar:

  def __init__(self, total, prefix='', suffix='', 
                    decimals=0, length=100, filling='█', empty='_'):
    """
    Creates terminal progress bar.

    Args:
      total   (int)           : a total number of iterations
      prefix  (str, optional) : prefix to progress bar
      suffix  (str, optional) : suffix of progress bar
      decimal (int, optional) : a number of decimals in percent complete
      length  (int, optional) : a bar length in characters
      fill    (str, optional) : the fillig of a progress bar
      empty   (str, optional) : an empty symbol in a progress bar
    """ 
    self.total = total - 1 # consider starting with 0
    self.prefix = prefix
    self.suffix = suffix
    self.decimals = decimals
    self.length = length
    self.filling = filling
    self.empty = empty
  
  def show(self, iteration):
    """Print and update a progress bar status; `iteration` is a current 
    iterations """
    percent = 100 * iteration / float(self.total)
    percent_str = f"{percent:0.{self.decimals}f}"
    len_filled = int(self.length * iteration // self.total)
    bar = self.filling * len_filled + self.empty * (self.length - len_filled)

    print(f'{self.prefix} |{bar}| {percent_str}% {self.suffix}', end='\r')
    # print a new line on complete
    if iteration == self.total:
      print()


class Id:

  __gen = None

  @staticmethod
  def next():
    if Id.__gen is None:
      Id.__gen = Id._new_gen(0)
    return next(Id.__gen)
  
  @staticmethod
  def reset(start=0):
    Id.__gen = Id._new_gen(start)
  
  @staticmethod
  def _new_gen(start):
    _id = start
    while True:
      yield _id
      _id += 1


class ID:
  __instance = None
  __gen = None
  __start = 0

  def __new__(cls):
    if ID.__instance is None:
      ID.__instance = super().__new__(cls)
    return ID.__instance
  
  def __init__(self, start=0):
    if ID.__gen is None:
      ID.__gen = self._new_gen(start)
      ID.__start = start

  def __next__(self):
    return next(ID.__gen)
  
  def reset(self, start=None):
    if start is not None:
      ID.__start = start
    ID.__gen = self._new_gen(start)

  def _new_gen(self, start):
    _id = start
    while True:
      yield _id
      _id += 1

if __name__ == "__main__":
  # from time import sleep
  # total = 100
  # bar = ProgressBar(total)
  # for i in range(total):
  #   sleep(4 / total)
  #   bar.show(i)

  # for _ in range(10):
  #   print(Id.next())
  
  # Id.reset(5)

  # print(Id.next())
  # print(Id.next())

  for _ in range(10):
    print(next(ID()))
  
  ID().reset(5)
  print(next(ID()))