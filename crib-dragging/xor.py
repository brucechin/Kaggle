
def sxor(s1,s2):
    # convert strings to a list of character pair tuples
    # go through each tuple, converting them to ASCII code (ord)
    # perform exclusive or on the ASCII code
    # then convert the result back to ASCII (chr)
    # merge the resulting array of characters as a string
    return ''.join(a ^ b for a,b in zip(s1,s2))

with open('citext0','rb') as f:
    data1 = f.read()

with open('citext1', 'rb') as f:
    data2 = f.read()
space = []
XORarray = []
five = []
with open('three.txt','r') as f:
    data3 = f.read()

three = data3.split(' ')

with open('four.txt','r') as f:
    data4 = f.read()

four = data4.split('\n')
Three = [i.capitalize() for i in three]
def cribdrag(cipher,word):
    l1 = len(word)
    l2 = len(cipher)
    for i in range(l2-l1):
        count = ''
        for j in range(l1):
            tmp =  cipher[i + j] ^ ord(word[j])
            if(tmp > 126 or tmp < 32) : break
            count += chr(tmp)
        if(count in four):
            print(i,end="")
            print("  ",end="")
            print(word, end="")
            print("  ", end="")
            print(count,end="")
            print()

def crib(cipher,word):
    l1 = len(word)
    count = ''
    for j in range(l1):
        tmp =  cipher[j] ^ ord(word[j])
        count += chr(tmp)
    print(i, end="")
    print("  ", end="")
    print(count,end="")
    print()

res =""

t = 0


with open('citext2.txt','w') as tt:
    for (i, j) in zip(data1, data2):
        tmp = ord(i) ^ ord(j)
        h = hex(tmp)[2:]
        if(len(h) == 1):
            h = '0' + h
        tt.write(h)
        print(h)

with open('words.txt','r') as words:
    wordslist = words.read()
    wordslist = wordslist.split('\n')

finalwordslist = []

for i in wordslist:
    if(len(i) > 2):
        tmp = ' '+ i + ' '
        print(tmp[:-1])
        print(len(tmp))
        finalwordslist.append(tmp)

print(finalwordslist)
#crib(XORarray,' ')
#print(space)

#for i in range(26):
#    tmp = ord('A') + i
#    tmp = tmp ^ ord(' ')
#    print(chr(tmp))