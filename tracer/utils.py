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

if __name__ == "__main__":
  from time import sleep
  total = 100
  bar = ProgressBar(total)
  for i in range(total):
    sleep(4 / total)
    bar.show(i)