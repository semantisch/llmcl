% https://daydreampuzzles.com/logic-puzzles/the-cheesy-chuckle-club/

%*
The Cheesy Chuckle Club is planning its next stand up comedy night with four slots on the line up. However, since a new comedy club opened up across the street, they've been struggling to book quality acts and have had to ask back four comedians whom they had previously banned! Can you match up each comedian with the time they will go on, the theme of their act, and the reason for their prior ban from the club?
*%

input(time,1,"7:00").
input(time,2,"7:30").
input(time,3,"8:20").
input(time,4,"9:00").

input(comedian,1,larry).
input(comedian,2,waylan).
input(comedian,3,abigail).
input(comedian,4,rip).

input(theme,1,cat).
input(theme,2,house).
input(theme,3,divorce).
input(theme,4,reunions).

input(reason,1,shoe).
input(reason,2,drunk).
input(reason,3,staff).
input(reason,4,tips).

index(time).

% 1) Abigail Later has a busy schedule so the club agreed to her request for the slot immediately after Larry Laughs.

clue(1,(less;next)).
object(1,1,comedian,larry).
object(1,2,comedian,abigail).
target(1,time).

% 2) The divorce act, which will be every bit as bitter as it sounds, will be later in the night than the family reunion act.

clue(2,less).
object(2,1,theme,reunions).
object(2,2,theme,divorce).
target(2,time).

% 3) The 7:30pm slot is either the divorce act or the comedian who was previously told not to return for insulting the club's staff.

clue(3,same).
object(3,1,time,"7:30").
object(3,2,theme,divorce).
object(3,2,reason,staff).

clue(4,diff).
object(4,1,theme,divorce).
object(4,2,reason,staff).

% 4) The comedian who has previously insulted the clubs staff is too foul tempered to enjoy the company of cats, and has no experience with which to speak on the subject. The person who will speak about their new kitty is not scheduled to go on at 8:20pm.

clue(5,diff).
object(5,1,reason,staff).
object(5,2,theme,cat).

clue(6,diff).
object(6,1,time,"8:20").
object(6,2,theme,cat).

% 5) Rip Van Roar will be performing immediately after the comedian who was previously banned due to turning up so drunk he or she couldn't climb the stairs to the stage. This was an embarrassing night for the club, as they didn't have a backup performer.

clue(7,(less;next)).
object(7,1,reason,drunk).
object(7,2,comedian,rip).
target(7,time).

% 6) The comedian who once stole the bartenders' tips is either Larry Laughs or the act that will be going on at 7:30pm. The brazen thief has since apologized to management, and is performing for free - immediately after Waylan Whoop.

clue(8,same).
object(8,1,reason,tips).
object(8,2,comedian,larry).
object(8,2,time,"7:30").

clue(9,diff).
object(9,1,comedian,larry).
object(9,2,time,"7:30").

clue(10,(less;next)).
object(10,1,comedian,waylan).
object(10,2,reason,tips).
target(10,time).
