%*
- LINUX call scheme:

  clingo --outf=1 logic-puzzles.lp ../instances/<instance>.lp | grep -v "^ANSWER" | clingo - check.lp ../instances/<instance>_sol.lp

- LINUX example call:

  clingo --outf=1 logic-puzzles.lp ../instances/daycare-drop-offs.lp | grep -v "^ANSWER" | clingo - check.lp ../instances/daycare-drop-offs_sol.lp
*%

%*
- WINDOWS call scheme:

  clingo --outf=1 logic-puzzles.lp ../instances/<instance>.lp | findstr /v "^ANSWER" | clingo - check.lp ../instances/<instance>_sol.lp

- WINDOWS example call:

  clingo --outf=1 logic-puzzles.lp ../instances/daycare-drop-offs.lp | findstr /v "^ANSWER" | clingo - check.lp ../instances/daycare-drop-offs_sol.lp
*%

:- match(X1,Y1,X2,Y2), not solution(X1,Y1,X2,Y2).
:- solution(X1,Y1,X2,Y2), not match(X1,Y1,X2,Y2).

#show match/4.
