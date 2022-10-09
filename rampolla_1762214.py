#Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")



#Python If-Else
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if (n%2==1): print("Weird")
    if (n%2==0 and 2<= n<= 5): print("Not Weird")
    if (n%2==0 and 6<=n <=20): print("Weird")
    if (n%2==0 and n>20): print("Not Weird")

#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)


#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)


#Loops
if __name__ == '__main__':
    n = int(input())
    for x in range(n):
        print(x*x)








#Print Function
if __name__ == '__main__':
    n = int(input())
    s = ''
    for x in range(n):
        s = s+str(x+1)
    print(s)







#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    arr=[]
    for a in range(x+1):
        for b in range(y+1):
            for c in range(z+1):
                if (a+b+c!=n):
                    arr.append([a,b,c])
    print(arr)

#Write a function
def is_leap(year):
    leap = False
    if (1900 <= year <= 10**5):
        if(year%4==0 and not(year%100==0)):
            leap = True
        elif(year%400==0 and year%100==0):
            leap= True
    
    return leap








#Find the Runner-Up Score!  
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    lista = sorted(set(arr))
    print(lista[-2])

#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    valori = student_marks.get(query_name)
    s = 0
    k = 0
    for x in valori:
        k += 1
        s = s + x
    print("{:.2f}".format(s/k))
        
        




#Lists
if __name__ == '__main__':
    N = int(input())
    l = []
    while (N > 0):
        stringa = input().split()
        if (stringa[0] == 'insert'):
            i = int(stringa[1])
            e = int(stringa[2])
            l.insert(i,e)   
        if (stringa[0] == 'print'):
            print(l)
        if (stringa[0] == 'remove'):
            x = int(stringa[1])
            for a in l:
                if a==x:
                    l.remove(a)
                    break
                    
        if (stringa[0] == 'append'):
            a = int(stringa[1])
            l.append(a)
        if (stringa[0] == 'sort'):
            l = sorted(l)
        if (stringa[0] == 'pop'):
            del l[-1]
        if (stringa[0] == 'reverse'):
            l.reverse()
        N = N - 1

#Tuples 
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    int_l = list(integer_list)
    l = []
    for x in range(n):
        l.append(int_l[x])
    print(hash(tuple(l)))

#sWAP cASE
def swap_case(s):
    return s.swapcase()



#String Split and Join


def split_and_join(line):
    line = line.split(" ")
    new = "-".join(line)
    return new

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    print("Hello " + first + " " +last +"! You just delved into python.")



#Mutations
def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]







#Find a string
def count_substring(string, sub_string):
    c = 0
    for x in range(0, len(string)):
        if sub_string[0] == string[x] and (x+len(sub_string) <= len(string)):
            t = True
            for y in range(0, len(sub_string)):
                if (sub_string[y]) != string[x+y]:
                    t = False
            if t: 
                c +=1
                
    return c




#String Validators
if __name__ == '__main__':
    x = input()
    isAlnum = False
    isAlpha = False
    isDigit = False
    isLower = False
    isUpper = False
    for s in x:
        if s.isalnum():
            isAlnum = True
        if s.isalpha():
            isAlpha = True 
        if s.isdigit():
            isDigit = True
        if s.islower():
             isLower = True 
        if s.isupper():
            isUpper = True
    print(isAlnum)
    
    print(isAlpha)
    
    print(isDigit)
    
    print(isLower)
    
    print(isUpper)

#Text Alignment
#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap


def wrap(string, max_width):
    arr = textwrap.wrap(string,max_width)
    return "\n".join(arr) 


#Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    s = input()
    n,m = map(int, s.split())
    i = 1
    j = n-2
    while i < n:
        print((i*".|.").center(m,"-"))
        i = i+2
    print(("WELCOME").center(m,"-"))
    while j > 0:
        print((j*".|.").center(m,"-"))
        j = j -2
    
    

#String Formatting
def print_formatted(number):
    for x in range(1,n+1):
        dec = str(x)
        octal = str(oct(x)[2:])
        hexadecimal = str(hex(x)[2:]).upper()
        binary = str(bin(x)[2:])
        lend = len(bin(number)[2:])-len(dec)
        leno = len(bin(number)[2:])-len(octal)
        lenh = len(bin(number)[2:])-len(hexadecimal)
        lenb = len(bin(number)[2:])-len(binary)
        print(lend*" "+dec,leno*" "+octal,lenh*" "+hexadecimal,lenb*" "+binary)



#Birthday Cake Candles
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
   maxc = max(candles)
   i = candles.count(maxc)
   return i

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()




#Capitalize!


# Complete the solve function below.
def solve(s):
    arr = s.split(" ")
    for x in range(len(arr)):
        arr[x] = arr[x].capitalize()
    return " ".join(arr)



#Merge the Tools!
def merge_the_tools(string, k):
    arr = [string[i:i+k] for i in range(0, len(string), k)]
    for parola in arr:
        ricorrenze=[]
        stri =''
        for char in parola:
            if char not in ricorrenze:
                stri +=char
            if (parola.count(char) > 1):
                ricorrenze.append(char)
        print(stri)
                    
                
    






#The Minion Game
def minion_game(string):
    stuart = 0
    kevin = 0
    arr = ["A","E","I","O","U"]
    vow = []
    con = []
    for x in range(len(string)):
        if string[x] in arr:
            kevin += len(string)-x
        else:
            stuart += len(string)-x
    
    if stuart>kevin:
        print("Stuart", stuart) 
    elif kevin>stuart:
        print("Kevin", kevin) 
    else: print("Draw")
        
        





#Introduction to Sets
def average(array):
    s = set(array)
    x = sum(s)/len(s)
    return round(x,3)


#No Idea!
if __name__ == '__main__':
    n,m = map(int, input().split())
    arr = input().split()
    A = set(input().split())
    B = set(input().split())
    happy = 0
    for x in arr:
        if (x in A):
            happy += 1
        if (x in B):
            happy -= 1
    print(happy)
         

#Set .add() 
if __name__ == '__main__':
    s = set()
    n = int(input())
    for x in range(0,n):
        s.add(input())
    print(len(s))

#Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
n = int(input())
for x in range(n):
    com = input().split()
    if com[0] == "discard":
        s.discard(int(com[1]))
    elif com[0] == "remove":
        s.remove(int(com[1]))
    else:
        s.pop()
print(sum(s))

#Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
print(len(s1.union(s2)))
    

#Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT


n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
print(len(s1.intersection(s2)))
    

#Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
print(len(s1.difference(s2)))
    

#Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT


n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
print(len(s1.symmetric_difference(s2)))
    

#Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
set1 = set(map(int, input().split()))
num = int(input())
for x in range(num):
    com = input().split()
    set2 = set(map(int, input().split()))
    if com[0] == "update":
        set1.update(set2)
    if com[0] == "intersection_update":
        set1.intersection_update(set2)
    if com[0] == "symmetric_difference_update":
        set1.symmetric_difference_update(set2)
    if com[0] == "difference_update":
        set1.difference_update(set2)
print(sum(set1))








#The Captain's Room 
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT


#easy way: dictionary
#but since this is in the category of sets i will try to use them without going in runtime error
n = int(input())
arr = list(map(int, input().split()))
dic = {}
for x in arr:
    if x not in dic:
        dic[x] = 1
    else:
        dic[x] += 1
        
for x in dic.keys():
    if dic[x] == 1:
        print(x)

#Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
set1 = set(map(int,input().split()))
N = int(input())
set2 = set(map(int,input().split()))

set3 = set1.difference(set2)
set4 = set2.difference(set1)
set5 = set3.union(set4)


for x in sorted(set5):
    print(x)

#Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
nt = int(input())

for x in range(nt):
    nA = int(input())
    sA = set(map(int, input().split()))
    nB = int(input())
    sB = set(map(int, input().split()))
    if len(sB.difference(sA)) == (nB-nA):
        print(True)
    else:
        print(False)

#Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
sA = set(map(int, input().split()))
nSet = int(input())
bool = True

for x in range(nSet):
    set1 = set(map(int, input().split()))
    if not set1.issubset(sA):
        bool = False
        break
print(bool)



#collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
X = int(input())
l = list(map(int, input().split()))
counter = Counter(l)
c = int(input())
money = 0
for x in range(c):
    s,p = map(int, input().split())
    if counter[s] != 0:
        money += p
        counter[s] -= 1
print(money)



#DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict

d = defaultdict(list)
n,m = map(int, input().split())
for x in range(n):
    word = input()
    d[word].append(str(x+1))
for y in range(m):
    word = input()
    if word in d:
        print(' '.join(d[word]))
    else:
        print("-1")

#Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple

n = int(input())
col_name = input().split()
Info = namedtuple('Info',[col_name[0],col_name[1],col_name[2],col_name[3]])
marks = 0
for x in range(n):
    row = input().split()
    r = Info(row[0],row[1],row[2],row[3],)
    marks += int(r.MARKS)
print(marks/n)

#Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict

n = int(input())
d = OrderedDict()
for x in range(n):
    s = input().split()
    price = int(s[-1])
    del s[-1]
    name = " ".join(s)
    if name not in d.keys():
        d[name] = price
    else:
        d[name] += price
    

for key, price in d.items():
    print(key, price)

#Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT

from collections import OrderedDict

n = int(input())
d = OrderedDict()
for x in range(n):
    s = input()
    if s not in d.keys():
        d[s] = 1
    else:
        d[s] += 1
print(len(d.keys()))
print(" ".join(map(str, d.values())))

#Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque

n = int(input())
d = deque()
for x in range(n):
    s = input().split()
    if s[0] == "append":
        d.append(int(s[1]))
    elif s[0] == "appendleft":
        d.appendleft(int(s[1]))
    elif s[0] == "pop":
        d.pop()
    else:
        d.popleft()
print(" ".join(map(str, d)))





#Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque


T = int(input())
for x in range(T):
    n = int(input())
    d =deque(map(int, input().split()))
    if d[0] > d[-1]:
        m = d[0]
        d.popleft()
    else:
        m = d[-1]
        d.pop()
    for x in range(n-1):
        if m >= d[0] and d[0] >= d[-1]:
            m = d[0]
            d.popleft()
        elif m >= d[-1] and d[-1] > d[0]:
            m = d[-1]
            d.pop()
    print("Yes") if len(d)==0 else print("No")
    
    
    

#Company Logo
#!/bin/python3

import math
import os
import random
import re
import sys

def find_max(l):
    c = ""
    max = 0
    for x in l:
        if x[1] > max:
            max = x[1]
            c = x[0]
        if x[1] == max:
            if x[0] < c:
                max = x[1]
                c = x[0]
    print(c, max)
    return c, max
    


if __name__ == '__main__':
    s = input()
    #i use the dictionary to count the occurence
    dic ={}
    for x in s:
        if x not in dic:
            dic[x] = 1
        else:
            dic[x] += 1
for x in range(3):
    l = list(dic.items())
    c, max = find_max(l)
    dic.pop(c)

    

#Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
dic = {'0':'MONDAY','1':'TUESDAY','2':'WEDNESDAY','3':'THURSDAY','4':'FRIDAY','5':'SATURDAY','6':'SUNDAY'}
m, d, y = map(int,input().split())
day = calendar.weekday(y,m,d)
print(dic[str(day)])



#Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())


for x in range(T):
    a,b = input().split()
    try:
        print(int(a)//int(b))
    except ZeroDivisionError as e:
        print("Error Code:",e)
    except ValueError as e:
        print("Error Code:", e)


#Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT

n,x = map(int, input().split())
l = []
for a in range(x):
    riga = list(map(float, input().split()))
    l.append(list(riga))
X = []
for b in l:
    X += [b]
sv = list(zip(*X))
for y in range(n):
    a = x
    b = sum(list(sv[y]))
    print(b/x)

#Input()
# Enter your code here. Read input from STDIN. Print output to STDOUT

x, k = map(int, input().split())
s = input()
if k == eval(s):
    print(True)
else:
    print(False)





#ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
def sort_letters(s):
    s2 = "".join(sorted(s))
    while s2[0].isupper():
        s2 += s2[0]
        s2 = s2.replace(s2[0],'',1)
    return s2

def sort_digits(d):
    odd =[]
    even = []
    s = ''
    for x in d:
        if x%2==0 or x == 0:
            even.append(x)
        else:
            odd.append(x)
    for x in odd:
        s += str(x)
    for x in even:
        s += str(x)
    return s
    
s = input()
letters = ''
digits = []
for x in s:
    if x.isalpha():
        letters += x
    else:
        digits.append(int(x))
letters_ord = sort_letters(letters)
digits.sort()
digits_ord = sort_digits(digits)
print(letters_ord+digits_ord)








#Map and Lambda Function
cube = lambda x: x**3

def fibonacci(n):
    l = []
    a = 0
    b = 1
    if n>=1:
        l.append(a)
    for x in range(n - len(l)):
        l.append(a+b)
        a = l[x]
        b = l[x+1]
    return l
        



#XML 1 - Find the Score


def get_attr_number(node):
    l = len(node.attrib)
    for x in node:
        l += get_attr_number(x)
    return l
        
        

#XML2 - Find the Maximum Depth


maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level > maxdepth:
        maxdepth = level
    for x in elem:
        depth(x, level)
        
    
            
        


#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        nl= []
        for x in l:
            if len(x)>= 10:
                if len(x) == 11:
                    x = "+91" + x[1:]
                if len(x) == 12:
                    x = "+91" + x[2:]
                if len(x) == 10:
                    x = "+91" + x
                nl.append(" ".join([x[:3],x[3:8],x[8:]]))
        l = f(nl)
    return fun




#Decorators 2 - Name Directory

from operator import itemgetter
from itertools import groupby
def person_lister(f):
    def inner(people):
        for x in people:
            x[2] = int(x[2])
        people.sort(key=itemgetter(2))
        return map(f,people)
    return inner


#Arrays


def arrays(arr):
    a = numpy.array(arr,float)
    return numpy.flip(a)


#Shape and Reshape
import numpy



arr = input().split()
nArr = numpy.array(arr,int)
nArr.shape = (3,3)
print(nArr)


#Transpose and Flatten
import numpy



n,m = map(int,input().split())
arr = []
for x in range(n):
    arr.append([x for x in map(int, input().split())])
nArr = numpy.array(arr, int)
print(numpy.transpose(nArr))
print(nArr.flatten())

#Concatenate
import numpy



n,m,p = map(int, input().split())
N =[([int(x) for x in input().split()] ) for i in range(n)]
P =[([int(x) for x in input().split()] ) for i in range(m)]
N = numpy.array(N)
P = numpy.array(P)
x = numpy.concatenate((N,P))
print(x)


#Zeros and Ones
import numpy


a = list(map(int, input().split()))
print(numpy.zeros(a, dtype = int))
print(numpy.ones(a, dtype= int))

#Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')


n,m = map(int, input().split())
print(numpy.eye(n,m,k=0))

#Array Mathematics
import numpy



n,m = map(int, input().split())
a = numpy.array([tuple([int(x) for x in input().split()]) for i in range(n)])
b = numpy.array([tuple([int(x) for x in input().split()]) for i in range(n)])
print(numpy.add(a,b))
print(numpy.subtract(a,b))
print(numpy.multiply(a,b))
print(a//b)
print(numpy.mod(a,b))
print(numpy.power(a,b))

#Floor, Ceil and Rint
import numpy

numpy.set_printoptions(legacy='1.13')

arr = list(map(float, input().split()))
nArr = numpy.array(arr)

print(numpy.floor(nArr))
print(numpy.ceil(nArr))
print(numpy.rint(nArr))

#Sum and Prod
import numpy



n,m = map(int,input().split())

nArr = [numpy.array([int(x) for x in input().split()]) for i in range(m)]
print(numpy.prod(numpy.sum(nArr, axis=0)))

#Min and Max
import numpy



n,m = map(int, input().split())

arr = [numpy.array([int(x) for x in input().split()]) for i in range(n)]
print(numpy.max(numpy.min(arr, axis=1)))

#Mean, Var, and Std
import numpy



n,m = map(int, input().split())

arr = [numpy.array([int(x) for x in input().split()]) for x in range(n)]

print(numpy.mean(arr, axis = 1))
print(numpy.var(arr, axis = 0))
print(round(numpy.std(arr, axis = None), 11))

#Dot and Cross
import numpy as np



n = int(input())

arrA = [np.array([int(x) for x in input().split()]) for _ in range(n)]
arrB = [np.array([int(x) for x in input().split()]) for x in range(n)]

print(np.dot(arrA,arrB))

#Inner and Outer
import numpy as np



arrA = np.array(list(map(int, input().split())))
arrB = np.array(list(map(int, input().split())))

print(np.inner(arrA, arrB))
print(np.outer(arrA, arrB))

#Polynomials
import numpy as np



P = np.array(list(map(float, input().split())))
x = int(input())
print(np.polyval(P,x))

#Linear Algebra
import numpy as np



n = int(input())
arr = [np.array([float(x) for x in input().split()]) for x in range(n)]

print(round(np.linalg.det(arr),2))







#Number Line Jumps
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    if (x2 > x1 and v2 > v1) or (x1 > x2 and v1 > v2):
        return "NO"
    elif v1 != v2 and ((x2-x1)%(v1-v2) == 0):
        return "YES"
    else:
        return "NO"
        
    
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()



#Viral Advertising
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def a(n):
    if n==1:
        return(math.floor(5/2), math.floor(5/2))
    b = a(n-1)
    return (b[0] + math.floor(b[1]*3/2), math.floor(b[1]*3/2))

def viralAdvertising(n):
    x,y = a(n)
    return x
   
    
    
    
    
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()








#Insertion Sort - Part 1
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#


def insertionSort1(n, arr):
    e = arr[-1]
    prec = arr[-2]
    while e < prec and n>1:
        arr[n-1] = prec
        n -= 1
        prec = arr[n-2]
        print(" ".join(map(str, arr)))
    arr[n-1] = e
    print(" ".join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort - Part 2
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    for x in range(n-1):
        a = x
        while arr[a] > arr[a+1] and a >= 0 and a < n-1:
            arr[a], arr[a+1] = arr[a+1], arr[a]
            a -= 1
        print(" ".join(map(str, arr)))
            

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)



#Recursive Digit Sum
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    s = sum(map(int, list(n)))*k
    if s > 10:
        s = superDigit(str(s), 1)
    return s

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()



#Detect Floating Point Number
import re
n = int(input())
for x in range(n):
    if re.match("^[+,-]?[0-9]*\.+[0-9]+$", input()):
        print(True)
    else:
        print(False)


#Re.split()
regex_pattern = r"[,\.]"	# Do not delete 'r'.



#Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT

import re

s = input()
m = re.search(r"([a-zA-Z0-9])\1+",s)

if m:
    print(m.group(1))
else:
    print(-1)




#Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT

import re
s= input()

m = re.findall(r"(?<=[^aeiouAEIOU])([aeiouAEIOU]{2,})(?=[^aeiouAEIOU])", s)
if m:
    for x in m:
        print(x)
else:
    print(-1)


#Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input()
k = input()

match = False
for x in range(len(S)):
    m = re.search(k,S[x:(x+len(k))])
    if m:
        match= True
        print((m.start()+x , m.end()+x-1))
if not match:
    print((-1,-1))
      

#Regex Substitution
import re

n = int(input())
s = [input() for x in range(n) ]

for x in range(len(s)):
    s[x] = re.sub(r"((?<= )(&&)(?= ))", "and", s[x])
    s[x] = re.sub(r"((?<= )(\|\|)(?= ))", "or", s[x])


print("\n".join(s))

#Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

#i had to look online at all the ROMAN NUMBERS and all the possibility

#Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
s = [input() for x in range(n)]

for x in s:
    print("YES") if re.match("^[789][0-9]{9}$",x) else print("NO")


#Validating and Parsing Email Addresses
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
s = [input() for x in range(n)]
for x in s:
    name = x.split()[0]
    mail = x.split()[1]
    if re.match(r"<[a-z][a-zA-Z0-9\-\.\_]+\@[a-zA-Z]+\.[a-zA-Z]{1,3}>",mail):
        print(x)

#Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
s = [input() for x in range(n)]


for x in range(len(s)):
    a = re.findall(r"#[0-9a-fA-F]{3,6}(?=[,|;|')'])", s[x])
    if a:
        for b in a:
            print(b)
    




#HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT

from html.parser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for x in attrs:
            print(f"-> {str(x[0])} > {str(x[1])}")
    def handle_endtag(self, tag):
        print(f"End   : {tag}")
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for x in attrs:
            print(f"-> {str(x[0])} > {str(x[1])}")

# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
n = int(input())
s = [input() for x in range(n)]
for x in s:
    parser.feed(x)



#HTML Parser - Part 2
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
import re


class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
          if re.search('\n',data):
              print (">>> Multi-line Comment", data, sep='\n')
          else:
              print (">>> Single-line Comment", data, sep='\n')
              
    def handle_data(self, data):
            if data != '\n':
                print (">>> Data", data, sep='\n')
        
        
n = int(input())
html = ""       
for _ in range(n):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)

#Detect HTML Tags, Attributes and Attribute Values
# Enter your code here. Read input from STDIN. Print output to STDOUT

from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrib):
        print(tag)
        for att in attrib:
            print(f"-> {att[0]} > {att[1]}")
            

n = int(input())
parser = MyHTMLParser()
s = "".join([input().strip() for x in range(n)])
parser.feed(s)


#Validating UID 
import re
n = int(input())
l = [input() for _ in range(n)]
for x in l:
    if re.match(r"^(?=(?:[a-z\d]*[A-Z]){2})(?=(?:\D*\d){3})(?:([a-zA-Z\d])(?!.*\1)){10}", x):
        print("Valid")
    else:
        print("Invalid")





#Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT

import re
n = int(input())
l = [input() for _ in range(n)]
for x in l:
    if re.match(r"^[456][0-9]{3}\-?[0-9]{4}\-?[0-9]{4}\-?[0-9]{4}$", x) and not re.search(r"(\d)\1{3}",x.replace("-","")):
        print("Valid")
    else:
        print("Invalid")


#Validating Postal Codes
regex_integer_in_range = r"(?=^[1-9]{1}\d{5}$)"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(?=(\d).\1)"	# Do not delete 'r'.



#Matrix Script
#!/bin/python3

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

s= "".join([matrix[i][j] for j in range(m) for i in range(n)])
print(re.sub(r'(?<=[a-zA-Z0-9])[^a-zA-Z0-9]+(?=[a-zA-Z0-9])',' ',s))


#Athlete Sort
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    m = int(first_multiple_input[1])

    arr = [list(map(int, input().rstrip().split())) for _ in range(n)]
    k = int(input().strip())
    arr =  sorted(arr, key= lambda x: x[k])
    for x in arr:
        print(*x)
        
    #we have also seen in the next exercises the "itemgetter" module, that would be also another possible solution. But we need to import it



#Time Delta
#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime as dt


# Complete the time_delta function below.
def time_delta(t1, t2):
    #solution with datetime
    f= "%a %d %b %Y %H:%M:%S %z"
    t1tm = dt.strptime(t1,f)
    t2tm = dt.strptime(t2,f)
    return str(round(abs((t1tm-t2tm).total_seconds())))
    
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()



#Nested Lists
if __name__ == '__main__':
    names = []
    scores = []
    for _ in range(int(input())):
        names.append(input())
        scores.append(float(input()))
    #i use the set to find the second lowest
    second = sorted(set(scores))[1]
    l = []
    for i in range(len(names)):
        if second == scores[i]:
            l.append(names[i])
    for name in sorted(l):
        print(name)
            
        
            


#Alphabet Rangoli
import string

def print_rangoli(size):
    a = " " + string.ascii_lowercase
    #the first loop is for the first half of the "pyramid", the second if 
    #for the lower half
    for x in range(size,0,-1):
        #the first string is for the left portion of each line of string
        st = a[size:x:-1]
        #while the second is for the right one
        st += a[x:size+1]
        #now i join the string with the char "-"
        s = "-".join(st)
        #then i print it with the center function 
        #and fill the empty spaces with "-"
        print(s.center(size*4-3,'-'))
    
        
    #same process as before, just x from min to max(size-1)
    for x in range(0,size-1):
        st = a[(size):x+2:-1]
        st += a[x+2:size+1]
        #first i join the 
        s = "-".join(st)
        print(s.center(size*4-3,'-'))



