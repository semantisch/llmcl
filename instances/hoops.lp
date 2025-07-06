% https://daydreampuzzles.com/logic-puzzles/hoops/

%*
Cody is a junior basketball coach and has just attended a district tournament where four of the teams he coaches participated. They all had hopes of getting to the state championships, but only two teams won their first game. Can you match each team with the name of its captain and their score from their first game? Note: The teams did not play against each other, and score alone does not indicate whether a team won their match as the scores of the opposing teams are not mentioned.
*%

input(team,1,giants).
input(team,2,breakers).
input(team,3,dribblers).
input(team,4,shot).

input(captain,1,alby).
input(captain,2,kyle).
input(captain,3,jericho).
input(captain,4,nisay).

input(score,1,46).
input(score,2,49).
input(score,3,51).
input(score,4,54).

index(team).

% 1) Kyle saved the game for his team (whose score didn't get past 50) with two free throws after the other team committed a foul.

clue(1,same).
object(1,1,captain,kyle).
object(1,2,score,(46;49)).

% 2) The Hills Giants, captained by Alby, is mostly comprised of a group of old friends who move together well on the court. Cody commented that this won them their game. They did not finish with a score of 51.

clue(2,same).
object(2,1,team,giants).
object(2,2,captain,alby).

clue(3,diff).
object(3,1,team,giants).
object(3,2,score,51).

% 3) Jericho's team scored higher than the Basket Breakers, but lower than Nisay's team.

clue(4,less).
object(4,1,team,breakers).
object(4,2,captain,jericho).
object(4,3,captain,nisay).
target(4,score).

% 4) Nice Shot lost their game after coming up against a very strong opposing team. Their final score was an even number.

clue(5,same).
object(5,1,team,shot).
object(5,2,score,(46;54)).

% 5) The Downtown Dribblers scored either 2 or 3 points more than Kyle's team.

clue(6,(less;next)).
object(6,1,captain,kyle).
object(6,2,team,dribblers).
target(6,score).
