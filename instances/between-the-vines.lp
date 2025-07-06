% https://daydreampuzzles.com/logic-puzzles/between-the-vines/

%*
Wine tour company "Between The Vines" is taking a group of 10 friends on a day trip to enjoy four local award winning wineries. From the clues provided, can you identify the order in which they will stop at each winery, the year the winery was established, and the variety of wine each winery recently won an award for?
*%

input(order,1,1).
input(order,2,2).
input(order,3,3).
input(order,4,4).

input(winery,1,bottling).
input(winery,2,cork).
input(winery,3,grapecious).
input(winery,4,vines).

input(award,1,tempranillo).
input(award,2,noir).
input(award,3,blanc).
input(award,4,shiraz).

input(year,1,1969).
input(year,2,1974).
input(year,3,1978).
input(year,4,1983).

index(order).

% 1) The four wineries are Vindictive Vines, the one who won an award for their Shiraz, the one established in 1969, and the one that will be visited 3rd.

clue(1,diff).
object(1,1,winery,vines).
object(1,2,award,shiraz).
object(1,3,year,1969).
object(1,4,order,3).

% 2) Of Chateau Cork and the winery established in 1974, one has the award winning Sauvignon Blanc and the other will be visited 2nd.

clue(2,diff).
object(2,1,winery,cork).
object(2,2,year,1974).

clue(3,same).
object(3,1,winery,cork).
object(3,1,year,1974).
object(3,2,award,blanc).

clue(4,same).
object(4,1,winery,cork).
object(4,1,year,1974).
object(4,2,order,2).

% 3) Boozy Bottling was not established between 1973 and 1977.

clue(5,diff).
object(5,1,winery,bottling).
object(5,2,year,1974).

% 4) The first winery to be visited won an award for a wine variety that has a single word name.

clue(6,same).
object(6,1,order,1).
object(6,2,award,(tempranillo;shiraz)).

% 5) The winery established in 1978 will be visited either immediately before or immediately after the winery with the award winning Shiraz.

clue(7,next).
object(7,1,year,1978).
object(7,2,award,shiraz).
target(7,order).

% 6) Of the winery who won an award for their Tempting Tempranillo and the winery known as Goodness Grapecious, one was established in 1978 and the other is the first destination of the wine tour.

clue(8,diff).
object(8,1,award,tempranillo).
object(8,2,winery,grapecious).

clue(9,same).
object(9,1,award,tempranillo).
object(9,1,winery,grapecious).
object(9,2,year,1978).

clue(10,same).
object(10,1,award,tempranillo).
object(10,1,winery,grapecious).
object(10,2,order,1).

% 7) The last winery on the tour was established 5 years after the first winery on the tour.

clue(12,same).
object(12,1,order,1).
object(12,2,year,(1969;1978)).

clue(13,(less;next)).
object(13,1,order,1).
object(13,2,order,4).
target(13,year).
