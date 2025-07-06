% https://daydreampuzzles.com/logic-puzzles/daycare-drop-offs/

%*
Dalton Daycare has 3 new attendees today, who have just been dropped off by their slightly anxious parents. Each child was given a toy from home to make them more comfortable. Using the clues provided, can you determine the age of each child and the toy they were dropped off with?
*%

input(name,1,max).
input(name,2,kylie).
input(name,3,emma).

input(age,1,3).
input(age,2,4).
input(age,3,5).

input(toy,1,train).
input(toy,2,dinosaur).
input(toy,3,dog).

index(name).

% 1) Max, the youngest of the three, does not have a plush dog toy.

clue(1,same).
object(1,1,name,max).
object(1,2,age,3).

clue(2,diff).
object(2,1,name,max).
object(2,2,toy,dog).

% 2) Kylie was comforted by her dinosaur toy.

clue(3,same).
object(3,1,name,kylie).
object(3,2,toy,dinosaur).

% 3) The 4 year old was not dropped off with a plush dog toy.

clue(4,diff).
object(4,1,age,4).
object(4,2,toy,dog).
