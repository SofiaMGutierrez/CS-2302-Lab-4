"""
Course: CS 2302
Author: Sofia Gutierrez
Lab #4: Compare the word embeddings of two given words devrived from
a list in order to compare the running times of BST and B-tree
Instructor: Olac Fuentes
T.A.: Anindita Nath
"""
import numpy as np
import time

class WordEmbedding(object): 
    def __init__(self,word,embedding=[]): 
        # word must be a string, embedding can be a list or and array of ints or floats 
        self.word = word 
        self.emb = np.array(embedding, dtype=np.float32) 
        # For Lab 4, len(embedding=50)

########################### BST ###########################

class BST(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

def InsertBST(T,newItem):
    if T == None:
        T =  BST(newItem)
    elif T.data.word > newItem.word:
        T.left = InsertBST(T.left, newItem)
    else:
        T.right = InsertBST(T.right, newItem)
    return T

def HeightBST(T):
    if T == None:
        return 0
    lh = HeightBST(T.left)
    rh = HeightBST(T.right)
    return max([lh,rh])+1

def NumOfNodesBST(T):
    if T == None:
        return 0
    else:
        count = 1
        count += NumOfNodesBST(T.left) + NumOfNodesBST(T.right)
    return count

def SearchBST(T, k):
    if T == None:
        return None
    elif k == T.data.word:
        return T
    elif k < T.data.word:
        return SearchBST(T.left, k)
    else:
        return SearchBST(T.right, k)

########################### BTree ###########################

class BTree(object):
    # Constructor
    def __init__(self,data,child=[],isLeaf=True,max_data=5):  
        self.data = data
        self.child = child 
        self.isLeaf = isLeaf
        if max_data <3: #max_data must be odd and greater or equal to 3
            max_data = 3
        if max_data%2 == 0: #max_data must be odd and greater or equal to 3
            max_data +=1
        self.max_data = max_data

def FindChild(T,k):
    # Determines value of c, such that k must be in subtree T.child[c], if k is in the BTree   
    for i in range(len(T.data)):
        if k.word < T.data[i].word:
            return i
    return len(T.data)

def InsertInternal(T, i):
    # T cannot be Full
    if T.isLeaf:
        InsertLeaf(T,i)
    else:
        k = FindChild(T,i)   
        if IsFull(T.child[k]):
            m, l, r = Split(T.child[k])
            T.data.insert(k,m) 
            T.child[k] = l
            T.child.insert(k+1,r) 
            k = FindChild(T,i)  
        InsertInternal(T.child[k],i)

def Split(T):
    #print('Splitting')
    #PrintNode(T)
    mid = T.max_data//2
    if T.isLeaf:
        leftChild = BTree(T.data[:mid],max_data=T.max_data) 
        rightChild = BTree(T.data[mid+1:],max_data=T.max_data) 
    else:
        leftChild = BTree(T.data[:mid],T.child[:mid+1],T.isLeaf,max_data=T.max_data) 
        rightChild = BTree(T.data[mid+1:],T.child[mid+1:],T.isLeaf,max_data=T.max_data) 
    return T.data[mid], leftChild,  rightChild

def InsertLeaf(T,i):
    T.data.append(i)
    T.data.sort(key = lambda x: x.word)

def IsFull(T):
    return len(T.data) >= T.max_data

def Insert(T,i):
    if not IsFull(T):
        InsertInternal(T,i)
    else:
        m, l, r = Split(T)
        T.data =[m]
        T.child = [l,r]
        T.isLeaf = False
        k = FindChild(T,i)
        InsertInternal(T.child[k],i)

def HeightBTree(T):
    if T.isLeaf:
        return 0
    return 1 + HeightBTree(T.child[0])

def NumOfNodesBTree(T):
    sum = len(T.data)
    for i in T.child:
        sum+=NumOfNodesBTree(i)
    return sum

def SearchBTree(T,k):
    for i in range(len(T.data)):
        if k.word == T.data[i].word:
            return T.data[i]
    if T.isLeaf: 
        return None
    return SearchBTree(T.child[FindChild(T,k)],k)

if __name__ == "__main__":
    
    print("1: Binary search tree")
    print("2: B-tree")
    menu_choice = int(input("Enter a menu option: "))
    
########################### BST ###########################
    if menu_choice == 1:
        
        print("Building binary search tree...")
        BST_T = None
        
        with open("glove.6B.50d.txt", "r", encoding='utf-8') as file:
            start1 = time.time()
            for line in file:
                line = line.split(" ")
                word_object = WordEmbedding(line[0], line[1:])
                BST_T = InsertBST(BST_T, word_object)
        end1 = time.time()
        
        print("Binary Search Tree stats:")
        print("Number of nodes:", NumOfNodesBST(BST_T))
        print("Height:", HeightBST(BST_T))
        print("Running time for binary search tree construction:", end1-start1)
        
        with open("words.txt", "r") as file: 
            start2 = time.time()
            for line in file:
                line = line.strip().split(" ")
                word1 = SearchBST(BST_T, line[0])
                word2 = SearchBST(BST_T, line[1])
                
                distance = np.dot(word1.data.emb,word2.data.emb)/(abs(np.linalg.norm(word1.data.emb))*abs(np.linalg.norm(word2.data.emb)))
                print("Similarity [", word1.data.word, word2.data.word, "] =", distance)
                
        end2 = time.time()
        
        print("Running time for binary search tree query processing: ", end2-start2)

########################### BTree ###########################
    if menu_choice == 2:
        
        user_max_data = int(input("Maximum number of items in node: "))
        print("Building B-tree...")
        BTree_T = BTree([], max_data = user_max_data)
        
        with open("glove.6B.50d.txt", "r", encoding='utf-8') as file:
            start1 = time.time()
            for line in file:
                list1 = line.split(" ")
                word_object = WordEmbedding(list1[0],list1[1:])
                Insert(BTree_T, word_object)
        end1 = time.time()
        
        print("B-tree stats:")
        print("Number of nodes:", NumOfNodesBTree(BTree_T))
        print("Height:", HeightBTree(BTree_T))
        print("Running time for B-tree construction:", end1-start1)
        
        with open("words.txt", "r") as file:
            start2 = time.time()
            for line in file:
                line = line.strip().split(" ")
                
                obj1 = WordEmbedding(line[0])
                obj2 = WordEmbedding(line[1])
                
                word1 = SearchBTree(BTree_T, obj1)
                word2 = SearchBTree(BTree_T, obj2)
                
                distance = np.dot(word1.emb,word2.emb)/(abs(np.linalg.norm(word1.emb))*abs(np.linalg.norm(word2.emb)))
                print("Similarity [", word1.word, word2.word, "] =", distance)
                
        end2 = time.time()
        
        print("Running time for B-tree query processing: ", end2-start2)