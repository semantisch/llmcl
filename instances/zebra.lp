% https://en.wikipedia.org/wiki/Zebra_Puzzle

input(house,1,1).
input(house,2,2).
input(house,3,3).
input(house,4,4).
input(house,5,5).

input(color,1,yellow).
input(color,2,blue).
input(color,3,red).
input(color,4,ivory).
input(color,5,green).

input(nationality,1,norwegian).
input(nationality,2,ukrainian).
input(nationality,3,englishman).
input(nationality,4,spaniard).
input(nationality,5,japanese).

input(drink,1,water).
input(drink,2,tea).
input(drink,3,milk).
input(drink,4,juice).
input(drink,5,coffee).

input(smoke,1,kools).
input(smoke,2,chesterfield).
input(smoke,3,old_gold).
input(smoke,4,lucky_strike).
input(smoke,5,parliament).

input(pet,1,fox).
input(pet,2,horse).
input(pet,3,snails).
input(pet,4,dog).
input(pet,5,zebra).

index(house).

% The Englishman lives in the red house.

clue(1,same).
object(1,1,nationality,englishman).
object(1,2,color,red).

% The Spaniard owns the dog.

clue(2,same).
object(2,1,nationality,spaniard).
object(2,2,pet,dog).

% Coffee is drunk in the green house.

clue(3,same).
object(3,1,drink,coffee).
object(3,2,color,green).

% The Ukrainian drinks tea.

clue(4,same).
object(4,1,nationality,ukrainian).
object(4,2,drink,tea).

% The green house is immediately to the right of the ivory house.

clue(5,(less;next)).
object(5,1,color,ivory).
object(5,2,color,green).
target(5,house).

% The Old Gold smoker owns snails.

clue(6,same).
object(6,1,smoke,old_gold).
object(6,2,pet,snails).

% Kools are smoked in the yellow house.

clue(7,same).
object(7,1,smoke,kools).
object(7,2,color,yellow).

% Milk is drunk in the middle house.

clue(8,same).
object(8,1,drink,milk).
object(8,2,house,3).

% The Norwegian lives in the first house.

clue(9,same).
object(9,1,nationality,norwegian).
object(9,2,house,1).

% The man who smokes Chesterfields lives in the house next to the man with the fox.

clue(10,next).
object(10,1,smoke,chesterfield).
object(10,2,pet,fox).
target(10,house).

% Kools are smoked in the house next to the house where the horse is kept.

clue(11,next).
object(11,1,smoke,kools).
object(11,2,pet,horse).
target(11,house).

% The Lucky Strike smoker drinks orange juice.

clue(12,same).
object(12,1,smoke,lucky_strike).
object(12,2,drink,juice).

% The Japanese smokes Parliaments.

clue(13,same).
object(13,1,nationality,japanese).
object(13,2,smoke,parliament).

% The Norwegian lives next to the blue house.

clue(14,next).
object(14,1,nationality,norwegian).
object(14,2,color,blue).
target(14,house).
