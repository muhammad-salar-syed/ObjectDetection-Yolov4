import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cvzone
import math

def findPokerHand(hand):
    ranks = []
    suits = []
    possibleRanks = []
 
    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
        if rank == "A":
            rank = 14
        elif rank == "K":
            rank = 13
        elif rank == "Q":
            rank = 12
        elif rank == "J":
            rank = 11
        ranks.append(int(rank))
        suits.append(suit)
 
    sortedRanks = sorted(ranks)
 
    # Royal Flush and Straight Flush and Flush
    if suits.count(suits[0]) == 5: # Check for Flush
        if 14 in sortedRanks and 13 in sortedRanks and 12 in sortedRanks and 11 in sortedRanks \
                and 10 in sortedRanks:
            possibleRanks.append(10)
        elif all(sortedRanks[i] == sortedRanks[i - 1] + 1 for i in range(1, len(sortedRanks))):
            possibleRanks.append(9)
        else:
            possibleRanks.append(6) # -- Flush
 
    # Straight
    # 10 11 12 13 14
    #  11 == 10 + 1
    if all(sortedRanks[i] == sortedRanks[i - 1] + 1 for i in range(1, len(sortedRanks))):
        possibleRanks.append(5)
 
    handUniqueVals = list(set(sortedRanks))
 
    # Four of a kind and Full House
    # 3 3 3 3 5   -- set --- 3 5 --- unique values = 2 --- Four of a kind
    # 3 3 3 5 5   -- set -- 3 5 ---- unique values = 2 --- Full house
    if len(handUniqueVals) == 2:
        for val in handUniqueVals:
            if sortedRanks.count(val) == 4:  # --- Four of a kind
                possibleRanks.append(8)
            if sortedRanks.count(val) == 3:  # --- Full house
                possibleRanks.append(7)
 
    # Three of a Kind and Pair
    # 5 5 5 6 7 -- set -- 5 6 7 --- unique values = 3   -- three of a kind
    # 8 8 7 7 2 -- set -- 8 7 2 --- unique values = 3   -- two pair
    if len(handUniqueVals) == 3:
        for val in handUniqueVals:
            if sortedRanks.count(val) == 3:  # -- three of a kind
                possibleRanks.append(4)
            if sortedRanks.count(val) == 2:  # -- two pair
                possibleRanks.append(3)
 
    # Pair
    # 5 5 3 6 7 -- set -- 5 3 6 7 - unique values = 4 -- Pair
    if len(handUniqueVals) == 4:
        possibleRanks.append(2)
 
    if not possibleRanks:
        possibleRanks.append(1)
    # print(possibleRanks)
    pokerHandRanks = {10: "Royal Flush", 9: "Straight Flush", 8: "Four of a Kind", 7: "Full House", 6: "Flush",
                      5: "Straight", 4: "Three of a Kind", 3: "Two Pair", 2: "Pair", 1: "High Card"}
    output = pokerHandRanks[max(possibleRanks)]
    print(hand, output)
    return output



classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']

np.random.seed(20)
COLORS = np.random.randint(0, 255, size=(len(classNames), 3), dtype="uint8")

load_model=torch.hub.load('ultralytics/yolov5','custom',path='./yolov5/yolov5/runs/train/exp2/weights/best.pt',force_reload=True)

video = cv2.VideoCapture('./poker.mp4')
ret, frame = video.read()
H, W, _ = frame.shape
# out = cv2.VideoWriter('./poker_v5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(video.get(cv2.CAP_PROP_FPS)), (W, H))

while ret:
    result=load_model(frame)
    hand = []
    for boxes in (result.xyxy[0]).tolist():

        x1,y1,x2,y2,conf,id=boxes[0],boxes[1],boxes[2],boxes[3],boxes[4],boxes[5]
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        Id = int(id)
        conf = math.ceil((conf * 100)) / 100
        #print(x1,y1,x2,y2,conf,Id)

        w, h = x2 - x1, y2 - y1
        # cvzone.cornerRect(img, (x1, y1, w, h))
        # cvzone.putTextRect(img, f'{classNames[Id]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
        color = [int(c) for c in COLORS[Id]]
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 3)
        cv2.putText(frame,f'{classNames[Id].upper()} {int(conf*100)}%',
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
        if conf > 0.5:
            hand.append(classNames[Id])

    hand = list(set(hand))
    
    if len(hand) == 5:
        results = findPokerHand(hand)
        #print(results)
        cvzone.putTextRect(frame, f'prediction: {results}', (40, 40), scale=2, thickness=2,colorT=(0, 255, 0), colorR=(0, 0, 0))
        
    cv2.imshow("Image", frame)
    cv2.waitKey(1)

    # out.write(frame)
    ret, frame = video.read()

# video.release()
# out.release()
# cv2.destroyAllWindows()