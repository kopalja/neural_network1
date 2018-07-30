import matplotlib.pyplot as plt
import fix_yahoo_finance as yf  




data = yf.download('MSFT', '2011-04-01', '2018-07-19')



#data.Close.plot()



open1 = data['Close']


print(open1)

x = [i for i in range(len(open1))]


plt.plot(x, open1)

plt.show()