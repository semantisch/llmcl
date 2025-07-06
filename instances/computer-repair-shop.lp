% https://daydreampuzzles.com/logic-puzzles/computer-repair-shop/

%*
Bigger Bytes Computer Repair Shop fixed five devices today, each for a local owner who insisted they knew what the issue was but turned out to be sorely mistaken! While not uncommon that the customers think they know best, today's reasons were some real doozies. Can you match each customer with the reason they claimed their device to be broken, and the real issue that was determined by the staff who fixed it?
*%

input(owner,1,claire).
input(owner,2,matthew).
input(owner,3,carlos).
input(owner,4,elka).

input(claimed,1,cursed).
input(claimed,2,warranty).
input(claimed,3,clicked).
input(claimed,4,color).

input(actual,1,hard_drive).
input(actual,2,memory).
input(actual,3,power).
input(actual,4,virus).

index(owner).

% 1) One customer, whose name begins with C, was adamant that their computer was suffering due to a friend recently installing a new mouse - red in color, unlike the black computer. Clearly incompatible! They were reluctant to believe the news that their computer actually need some bad memory replaced.

clue(1,same).
object(1,1,owner,(claire;carlos)).
object(1,2,claimed,color).
object(1,3,actual,memory).

% 2) The person whose device had a hard drive error was not the same person who stormed into the shop upset that their computer had been cursed by their unfriendly new neighbor.

clue(2,diff).
object(2,1,claimed,cursed).
object(2,2,actual,hard_drive).

% 3) Carlos was embarrassed to learn he had actually had his computer infected with a virus.

clue(3,same).
object(3,1,owner,carlos).
object(3,2,actual,virus).

% 4) The four unique customers were: Claire, the one who believed their computer was programmed to have an issue the moment the warranty expired, the one who believed they broke their computer by accidentally triple clicking instead of double clicking, and the one whose device was suffering a virus Infection.

clue(4,diff).
object(4,1,owner,claire).
object(4,2,claimed,(warranty;clicked)).
object(4,3,actual,virus).

% 5) Matthew's issue was not the power fault.

clue(5,diff).
object(5,1,owner,matthew).
object(5,2,actual,power).

% 6) The one who suffered the hard drive error was not the customer who believed their technical woes were a result of mistakenly triple clicking instead of double clicking.

clue(6,diff).
object(6,1,claimed,clicked).
object(6,2,actual,hard_drive).
