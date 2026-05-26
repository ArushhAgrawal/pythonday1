list={"name":"unknown", "score":0}
highest=0
q=0
r=0
topper=''
students=[]
while True:
    i= input("enter student name: ")  
    if i=="exit":
        break  
    q+=1
    list["name"]=i
    s= input("enter student score: ")
    list["score"] =int(s)
    if list["score"]>highest:
        highest=list["score"]
        topper=i
    students.append({**list})   
for e in range(q):
    print("names:", students[r]["name"])
    r+=1
print(topper + " congrats for scoring best score")
print("ur score is:", highest)
print(students)
