import matplotlib.pyplot as plt
import seaborn as sns

#fonctions de visualisation
def boxplot(data, x,y):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Survived', y='Age', data=data)
    plt.title("Boxplot de " + str(y) +" en fonction de " + str(x))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    
def distribution(data, x):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[x], kde=True)
    plt.title("Distribution de " + str(x))
    plt.xlabel(x)
    plt.ylabel("Frequency")
    plt.show()
    
def histogram(data, x):
    sns.histplot(data[x], bins=range(data[x].min(), data[x].max()+2), kde=False)
    plt.ylabel('Count')
    plt.xlabel(x)
    plt.title('Histogramme de la variable ' + str(x))
    plt.show()
    
    

