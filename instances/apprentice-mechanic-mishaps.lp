% https://daydreampuzzles.com/logic-puzzles/apprentice-mechanic-mishaps/

%*
Four apprentice mechanics work at Rusty's Repair Shop. They are generally well-regarded by management, but last week with less supervision there were some mistakes that led to a number of angry customers. Using the clues provided, can you determine which apprentice worked on which car, and the mistake that was made?
*%

input(name,1,jimmy).
input(name,2,kayla).
input(name,3,bart).
input(name,4,harrison).

input(car,1,civic).
input(car,2,corolla).
input(car,3,forester).
input(car,4,sierra).

input(mistake,1,hole).
input(mistake,2,cap).
input(mistake,3,stains).
input(mistake,4,problem).

index(name).

% 1) Bart didn't work on the Sierra, nor did he leave the oil cap off.

clue(1,diff).
object(1,1,name,bart).
object(1,2,car,sierra).
object(1,2,mistake,cap).

% 2) The four apprentices are Kayla, the one who worked on the Honda Civic, the one who left the oil cap off, and the one who left stains on the carpet.

clue(2,diff).
object(2,1,name,kayla).
object(2,2,car,civic).
object(2,3,mistake,(cap;stains)).

% 3) The receptionist was in tears after dealing with the customer who had found a hole was accidentally drilled in their gas tank (oops!). This wasn't the Sierra.

clue(3,diff).
object(3,1,mistake,hole).
object(3,2,car,sierra).

% 4) Jimmy didn't leave the oil cap off nor leave stains on the carpet of the vehicle he worked on.

clue(4,diff).
object(4,1,name,jimmy).
object(4,2,mistake,(cap;stains)).

% 5) Harrison worked on the Forester, and disliked every minute of it.

clue(5,same).
object(5,1,name,harrison).
object(5,2,car,forester).
