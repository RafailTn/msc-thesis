% dimscthesis.cls
%
% A LaTeX2e document class for writing MSc theses in the Department of
% Informatics and Telecommunications (DI&T) of the National and Kapodistrian
% University of Athens (NKUA).
%
% Copyright (c) 2023 Laertis Moustakas <lmoustak@gmail.com>
%
% Based on the diphdtesis.cls document class, Version 1.1, by
% Charalampos S. Nikolaou <charnik@di.uoa.gr>
%
% This work may be distributed and/or modified under the conditions of the
% LaTeX Project Public License. The latest version of this license is in
%   http://www.latex-project.org/lppl.txt.
%
% This work consists of the following files:
% dimscthesis.cls
%		This document class file
%
% thesis_en.tex
%		A sample document demonstrating the use of this class file
%
% emblems/
%		A directory containing three colored versions of the official emblem of
%		the National and Kapodistrian University of Athens.
%
% fonts/
%		A directory containing the Arial family TrueType fonts. The directory
%		also includes a README file with instructions on installing the fonts
%		in your system (assuming Unix-based).

% Document Class Options
%
% en
% 		If present, the document adds some English pages in the front matter
% 		plus changes some heading to English. Use this option if thesis is in English.
%       Omit if written in Greek.
%
% inscr
% 		If present, then a page with the inscription provided via the command
% 		\inscription{} is printed.
%
% ack
%		If present, then a page with the acknowledgements provided via the
%		command \acksEn{} is printed.
%
% preface
%		If present, then a preface page is included just before the
%		introductory chapter. The content of this page is controled via the
%		command \preface{}.
%
% oneside
%		If present, the thesis is formatted as if it will be printed on
%		only one side of each page, the other side being left blank.
%		Default is "twoside", which is used when this option is omitted.
%

% Document Versions
%
% Version 1.1, 2023/10/04
%		Additional improvements.
%
% Version 1.0, 2023/12/08
%		Initial attempt at creating the first class file for LaTeX Theses in
%		DI&T, NKUA.

% Known Issues
%
% * Chapter titles are not appearing in capital letters in the ToC, although
%   in other places this has been taken care of for good. Thus, one is enforced
%   to type chapter titles in capital letters, so as to be compatible with the
%   requirements of the department.
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dimscthesis}[2023/10/04 v1.1 LaTeX class for MSc Theses
submitted to the Department of Informatics and Telecommunications of the
National and Kapodistrian University of Athens (by Laertis Moustakas
<lmoustak@gmail.com>)]

%
% Declare and initialize global ifs
% (set by passing options to this document class)
%

% controls the main language being set to English
\newif\ifenglishlanguage
\englishlanguagefalse

% controls the inclusion of the inscription page
\newif\ifinscriptionpage
\inscriptionpagefalse

% controls inclusion of acknowledgement page
\newif\ifackpage
\ackpagefalse

% controls inclusion of the list of publications page
\newif\ifloppage
\loppagefalse

% controls inclusion of the preface page
\newif\ifprefacepage
\prefacepagefalse

%
% Declare and initialize internal ifs (not set by document class options)
%
\newif\ifenglishfrontpage
\englishfrontpagefalse

\newif\ifenglishcommitteepage
\englishcommitteepagefalse

\newif\ifoneside
\onesidefalse

% marks whether the appendices have started
% (used later for setting an appropriate naming scheme for the appendix title)
\newif\ifappendixstarted
\appendixstartedfalse

%
% Check passed options
%
\DeclareOption{en}{
  \englishlanguagetrue
}
\DeclareOption{inscr}{
  \inscriptionpagetrue
}
\DeclareOption{ack}{
  \ackpagetrue
}
\DeclareOption{preface}{
  \prefacepagetrue
}
\DeclareOption{oneside}{
  \onesidetrue
  \PassOptionsToClass{oneside}{book}
}


\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax

\LoadClass[12pt,openright]{book}

%
% Required packages and configuration
%
\RequirePackage[a4paper,
top=2cm,bottom=2cm,bindingoffset=0.5cm,left=2cm,right=2cm,
headheight=15pt,headsep=1.25cm,footskip=1.25cm,
includehead,includefoot]{geometry} % add option showframe=true for debugging
\RequirePackage{fancyhdr}
\RequirePackage{emptypage}
\RequirePackage[final]{graphicx}
\RequirePackage{sectsty}   % needed for formatting chapter headings
\RequirePackage{textcase}  % needed for forcing capital letters (in chapters)
\RequirePackage[resetlabels]{multibib}
%\usepackage[notlof,notlot,nottoc,notlop]{tocbibind}
\RequirePackage[titles]{tocloft}  % needed to adjust Toc margins and fonts
\RequirePackage{etoolbox}  % needed for patchcmd
\RequirePackage{tabularx}  % needed for tabular* environment
\RequirePackage{longtable} % needed for splitting big tables across pages
\RequirePackage{booktabs}
\RequirePackage{hyperref}
\RequirePackage[font={bf,footnotesize}]{caption}
\RequirePackage{amssymb}
\RequirePackage[english,greek]{babel}

%(e.g., abbreviations)
%\RequirePackage[toc,page,titletoc]{appendix} % needed for configuring
%%appendices

%
% Add Greek support
%
\usepackage{fontspec}
\usepackage{xunicode}
\usepackage{xltxtra}

%
% Set font family Arial
%
\setmainfont[Ligatures={Common,TeX}]{Arial} % the passed option was needed for
                                            % correctly rendering double quotes
\defaultfontfeatures{Mapping=tex-text,Scale=MatchLowercase}
\setsansfont[Mapping=tex-text,Scale=MatchLowercase]{Arial}
\setmonofont[Scale=0.95]{Courier New}

% Font sizes for a document class with a base size of 12pt:
%
% \tiny         = 6pt
% \scriptsize   = 8pt (footnotes)
% \footnotesize = 10pt (references at the last page)
% \small        = 10.95pt
% \normalsize   = 12pt (mostly throughout the document)
% \large        = 14.4pt
% \Large        = 17.28pt (title)
% \huge         = 24.88pt
% \Huge         = 24.88pt

%\ifglossaryInPreamble
%\RequirePackage[toc,section=chapter,numberedsection=false,nonumberlist]{glossaries}
%\else
% \RequirePackage[toc,section,numberedsection=nolabel,nonumberlist]{glossaries}
%\fi
%\makeglossaries

%
% Command for the program (greek and english)
%
\providecommand{\programGr}[1]{\gdef\@programGr{#1}}
\providecommand{\programEn}[1]{\gdef\@programEn{#1}}

%
% Commands for first, middle, last name, and serial number (greek and english versions)
%
\providecommand{\authorFirstGr}[1]{\gdef\@authorFirstGr{#1}}
\providecommand{\authorFirstAbrGr}[1]{\gdef\@authorFirstAbrGr{#1}}
\providecommand{\authorMiddleGr}[1]{\gdef\@authorMiddleGr{#1}}
\providecommand{\authorLastGr}[1]{\gdef\@authorLastGr{#1}}
\providecommand{\authorFirstEn}[1]{\gdef\@authorFirstEn{#1}}
\providecommand{\authorFirstAbrEn}[1]{\gdef\@authorFirstAbrEn{#1}}
\providecommand{\authorMiddleEn}[1]{\gdef\@authorMiddleEn{#1}}
\providecommand{\authorLastEn}[1]{\gdef\@authorLastEn{#1}}
\providecommand{\authorSN}[1]{\gdef\@authorSN{#1}}

%
% Commands for the title of the thesis (greek and english versions)
%
\providecommand{\titleGr}[1]{\gdef\@titleGr{#1}}
\providecommand{\titleEn}[1]{\gdef\@titleEn{#1}}

%
% Commands for the date of the writing of the thesis (Month followed by Year)
% [provide greek and english versions]
%
\providecommand{\dateGr}[1]{\gdef\@dateGr{#1}}
\providecommand{\dateEn}[1]{\gdef\@dateEn{#1}}


%
% Commands for advisor
%
\providecommand{\advisorGr}[1]{\gdef\@advisorGr{#1}}
\providecommand{\advisorRankGr}[1]{\gdef\@advisorRankGr{#1}}
\providecommand{\advisorOrgGr}[1]{\gdef\@advisorOrgGr{#1}}
\providecommand{\advisorEn}[1]{\gdef\@advisorEn{#1}}
\providecommand{\advisorRankEn}[1]{\gdef\@advisorRankEn{#1}}
\providecommand{\advisorOrgEn}[1]{\gdef\@advisorOrgEn{#1}}

%
% Commands for the 1st member of the advisory board
%
\newcommand{\boardOneGr}{\advisorGr}
\newcommand{\boardOneRankGr}{\advisorRankGr}
\newcommand{\boardOneOrgGr}{\advisorOrgGr}
\newcommand{\boardOneEn}{\advisorEn}
\newcommand{\boardOneRankEn}{\advisorRankEn}
\newcommand{\boardOneOrgEn}{\advisorOrgEn}

%
% Commands for the 2nd member of the examination board
%
\providecommand{\boardTwoGr}[1]{\gdef\@boardTwoGr{#1}}
\providecommand{\boardTwoRankGr}[1]{\gdef\@boardTwoRankGr{#1}}
\providecommand{\boardTwoOrgGr}[1]{\gdef\@boardTwoOrgGr{#1}}
\providecommand{\boardTwoEn}[1]{\gdef\@boardTwoEn{#1}}
\providecommand{\boardTwoRankEn}[1]{\gdef\@boardTwoRankEn{#1}}
\providecommand{\boardTwoOrgEn}[1]{\gdef\@boardTwoOrgEn{#1}}

%
% Commands for the 3rd member of the examination board
%
\providecommand{\boardThreeGr}[1]{\gdef\@boardThreeGr{#1}}
\providecommand{\boardThreeRankGr}[1]{\gdef\@boardThreeRankGr{#1}}
\providecommand{\boardThreeOrgGr}[1]{\gdef\@boardThreeOrgGr{#1}}
\providecommand{\boardThreeEn}[1]{\gdef\@boardThreeEn{#1}}
\providecommand{\boardThreeRankEn}[1]{\gdef\@boardThreeRankEn{#1}}
\providecommand{\boardThreeOrgEn}[1]{\gdef\@boardThreeOrgEn{#1}}

%
% Commands for setting up the dates for the examination
%
\providecommand{\examinationDateGr}[1]{\gdef\@examinationDateGr{#1}}
\providecommand{\examinationDateEn}[1]{\gdef\@examinationDateEn{#1}}

%
% Commangs for the supervisors section, including headings and people involved
%
\providecommand{\supervisorsGr}[2][Επιβλέπων]{
    \gdef\@supervisorHeadingGr{#1}
    \gdef\@supervisorsGr{#2}
}
\providecommand{\supervisorsEn}[2][Supervisor]{
    \gdef\@supervisorHeadingEn{#1}
    \gdef\@supervisorsEn{#2}
}

%
% Commands for setting up abstract (greek and english versions), ack, incsription, and preface pages
%
\providecommand{\abstractEn}[1]{\gdef\@abstractEn{#1}}
\providecommand{\abstractGr}[1]{\gdef\@abstractGr{#1}}
\providecommand{\acks}[1]{\gdef\@acknowledgements{#1}}
\providecommand{\inscription}[1]{\gdef\@inscription{%
\vspace*{0.2\textheight}
\begin{flushright}
#1
\end{flushright}
}}
\providecommand{\preface}[1]{\gdef\@preface{#1}}

%
% Commands for Subject Area and Keywords fields
% (greek and english versions)
%
\providecommand{\subjectAreaGr}[1]{\gdef\@subjectAreaGr{#1}}
\providecommand{\subjectAreaEn}[1]{\gdef\@subjectAreaEn{#1}}
\providecommand{\keywordsGr}[1]{\gdef\@keywordsGr{#1}}
\providecommand{\keywordsEn}[1]{\gdef\@keywordsEn{#1}}

%
% Command for starting a new unumbered chapter (ToC'ed) for
% holding the table of abbreviations and acronyms. The table
% should be set by the user.
%
\providecommand{\abbreviations}{
  \ifenglishlanguage
    \chapter*{ABBREVIATIONS - ACRONYMS}
    \addcontentsline{toc}{chapter}{ABBREVIATIONS - ACRONYMS}
  \else
    \chapter*{ΣΥΝΤΜΗΣΕΙΣ – ΑΡΚΤΙΚΟΛΕΞΑ – ΑΚΡΩΝΥΜΙΑ}
    \addcontentsline{toc}{chapter}{ΣΥΝΤΜΗΣΕΙΣ – ΑΡΚΤΙΚΟΛΕΞΑ – ΑΚΡΩΝΥΜΙΑ}
  \fi
}

%
% Front page (greek and english versions)
%
\def\@frontpage{
\begin{center}
\includegraphics{emblems/athena-black}
\end{center}
\begin{minipage}[t]{\textwidth}
    \begin{center}
      {\large \bfseries
      \ifenglishfrontpage
      	NATIONAL AND KAPODISTRIAN UNIVERSITY OF ATHENS
      \else
      	ΕΘΝΙΚΟ ΚΑΙ ΚΑΠΟΔΙΣΤΡΙΑΚΟ ΠΑΝΕΠΙΣΤΗΜΙΟ ΑΘΗΝΩΝ
      \fi
      }
      \linebreak

      {\bfseries
      \ifenglishfrontpage
      	SCHOOL OF SCIENCES \\ DEPARTMENT OF INFORMATICS AND TELECOMMUNICATIONS
      \else
      	ΣΧΟΛΗ ΘΕΤΙΚΩΝ ΕΠΙΣΤΗΜΩΝ \\ ΤΜΗΜΑ ΠΛΗΡΟΦΟΡΙΚΗΣ ΚΑΙ ΤΗΛΕΠΙΚΟΙΝΩΝΙΩΝ
      \fi
      }
      \linebreak

      {\bfseries
      \ifenglishfrontpage
      	\@programEn
      \else
      	\@programGr
      \fi}
      \linebreak\linebreak\linebreak
      \linebreak\linebreak\linebreak

      {\bfseries
      \ifenglishfrontpage
      	MSc THESIS
      \else
      	ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ
      \fi}
      \linebreak\linebreak\linebreak

      {\Large \bfseries
      \ifenglishfrontpage
      	\@titleEn
      \else
      	\@titleGr
      \fi}
      \linebreak\linebreak\linebreak

      {\bfseries
      \ifenglishfrontpage
	  	\@authorFirstEn{} \@authorMiddleEn{} \@authorLastEn
	  \else
	  	\@authorFirstGr{} \@authorMiddleGr{} \@authorLastGr
	  \fi}
    \end{center}
\end{minipage}
    \vfill
    \ifenglishfrontpage
    {\bfseries \@supervisorHeadingEn:}
    \begin{minipage}[t]{\textwidth}
        \@supervisorsEn
    \end{minipage}
    \else
    {\bfseries \@supervisorHeadingGr:}
    \begin{minipage}[t]{\textwidth}
        \@supervisorsGr
    \end{minipage}
    \fi
    \\

    \begin{center}
    {\bfseries
     \ifenglishfrontpage
     	ATHENS
     \else
	 	ΑΘΗΝΑ
     \fi
    }
    \linebreak

    {\bfseries
     \ifenglishfrontpage
     	\@dateEn
     \else
     	\@dateGr
     \fi
     }
    \end{center}
    \clearpage

    \newpage\null\newpage
}

%
% Committee page (greek and english versions)
%
\def\@committeepage{
\begin{center}
{\bfseries
\ifenglishcommitteepage
	MSc THESIS
\else
	ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ
\fi
}
\linebreak

{
\ifenglishcommitteepage
	\@titleEn
\else
	\@titleGr
\fi
}
\linebreak

{\bfseries
\ifenglishcommitteepage
	\@authorFirstEn{} \@authorMiddleEn{} \@authorLastEn\\
    S.N.:
\else
	\@authorFirstGr{} \@authorMiddleGr{} \@authorLastGr\\
    Α.Μ.:
\fi
}
\@authorSN
\linebreak\linebreak
\end{center}

\vfill
{
\ifenglishcommitteepage%
    {\bfseries \MakeUppercase{\@supervisorHeadingEn}:}
    \begin{minipage}[t]{\textwidth}
        \@supervisorsEn
    \end{minipage}
\else
    {\bfseries \MakeUppercase{\@supervisorHeadingGr}:}
    \begin{minipage}[t]{\textwidth}
        \@supervisorsGr
    \end{minipage}
\fi
}
%\linebreak\linebreak
%\linebreak

\vfill
{
\ifenglishcommitteepage
	{\bfseries EXAMINATION COMMITTEE:}
    \begin{minipage}[t]{\textwidth}
        {\bfseries \@advisorEn{}}, \@advisorRankEn{} \@advisorOrgEn{}\\
        {\bfseries \@boardTwoEn{}}, \@boardTwoRankEn{} \@boardTwoOrgEn{}\\
        {\bfseries \@boardThreeEn{}}, \@boardThreeRankEn{} \@boardThreeOrgEn{}
    \end{minipage}
\else
	{\bfseries ΕΞΕΤΑΣΤΙΚΗ ΕΠΙΤΡΟΠΗ:}
    \begin{minipage}[t]{\textwidth}
        {\bfseries \@advisorGr{}}, \@advisorRankGr{} \@advisorOrgGr{}\\
        {\bfseries \@boardTwoGr{}}, \@boardTwoRankGr{} \@boardTwoOrgGr{}\\
        {\bfseries \@boardThreeGr{}}, \@boardThreeRankGr{} \@boardThreeOrgGr{}
    \end{minipage}
\fi
}

\vfill
\begin{center}
{\bfseries
\ifenglishcommitteepage
	Examination Date: \@examinationDateEn
\else
	Ημερομηνία Εξέτασης: \@examinationDateGr
\fi
}
\end{center}
\clearpage
\newpage\null\newpage
}

%
% Abstract (english version)
%
\def\@absEn{
  \chapter*{Abstract}
  \thispagestyle{empty}
  \selectlanguage{english}
  \@abstractEn{}
  \vfill

  {\bfseries SUBJECT AREA}: \@subjectAreaEn\\

  {\bfseries KEYWORDS}: \@keywordsEn
}

%
% Abstract (greek version)
%
\def\@absGr{
  \chapter*{ΠΕΡΙΛΗΨΗ}
  \thispagestyle{empty}
  \selectlanguage{greek}
  \@abstractGr{}
  \vfill

  {\bfseries ΘΕΜΑΤΙΚΗ ΠΕΡΙΟΧΗ}: \@subjectAreaGr\\

  {\bfseries ΛΕΞΕΙΣ ΚΛΕΙΔΙΑ}: \@keywordsGr

  \clearpage
}

%
% Inscription page (optional)
%
% Enable it by passing option ``inscr'' to the document class.
%
\def\@inscr{
  \cleardoublepage
  \@inscription{}
  \clearpage
}

%
% Acknowledgements (optional)
%
% Enable it by passing option ``ack'' to the document class.
%
\def\@acks{
  \ifenglishlanguage
    \chapter*{Acknowledgments}
  \else
    \chapter*{ΕΥΧΑΡΙΣΤΙΕΣ}
  \fi
  \thispagestyle{empty}
  \@acknowledgements{}
  \clearpage
}

%
% Preface page (optional)
%
% Enable it by passing option ``preface'' to the document class.
%
% Check the following page that gives a definition among the uses of Prologue,
% Foreword, and Preface:
%http://iankingsleyauthor.blogspot.de/2013/03/defined-prologue-epilogue-foreword.html.
%
\def\@prefacepage{
  \ifenglishlanguage
    \chapter*{Preface}
  \else
    \chapter*{ΠΡΟΛΟΓΟΣ}
  \fi
  \thispagestyle{empty}
  \@preface{}
  \clearpage
}

%
% Modify Table of Content's indentations and font
%
\setlength{\cftsecindent}{0em}
\setlength{\cftsubsecindent}{0.42cm}
\setlength{\cftsubsubsecindent}{0.84cm}
\renewcommand{\cftsecfont}{\bfseries\footnotesize}
\renewcommand{\cftsecpagefont}{\bfseries\footnotesize}
\renewcommand{\cftsubsecfont}{\footnotesize}
\renewcommand{\cftsubsecpagefont}{\footnotesize}
\renewcommand{\cftsubsubsecfont}{\footnotesize}
\renewcommand{\cftsubsubsecpagefont}{\footnotesize}
\renewcommand{\cftbeforechapskip}{18pt}
\renewcommand{\cftbeforesecskip}{12pt}

%
% Configure the frontmatter page
%
\renewcommand{\frontmatter}{
    \pagestyle{empty}
    \ifenglishlanguage
        % frontpage - english version
        \englishfrontpagetrue
        \@frontpage
    \fi

    % frontpage - greek version
    \englishfrontpagefalse
    \@frontpage

    \ifenglishlanguage
        % examination committe page - english version
        \englishcommitteepagetrue
        \@committeepage
    \fi

    % examination committe page - greek version
    \englishcommitteepagefalse
    \@committeepage

    % abstract (greek and english version)
    \ifenglishlanguage
        \@absEn
        \@absGr
    \else
        \@absGr
        \@absEn
    \fi

    \ifenglishlanguage
        \selectlanguage{english}
    \else
        \selectlanguage{greek}
    \fi

    \ifinscriptionpage
        \@inscr
    \fi

    % acknowledgements
    \ifackpage
        \@acks
    \fi

    % table of contents
    % add TOC as bookmark
    \ifenglishlanguage
        \addtocontents{toc}{\protect{\pdfbookmark[0]{CONTENTS}{toc}}}
    \else
        \addtocontents{toc}{\protect{\pdfbookmark[0]{ΠΕΡΙΕΧΟΜΕΝΑ}{toc}}}
    \fi
    \tableofcontents

    % list of figures
    %\listoffigures
    %\thispagestyle{empty}
    \cleardoublepage
    \begingroup
    \makeatletter
    \let\ps@plain\ps@empty
    \makeatother

    \pagestyle{empty}
    \listoffigures
    \cleardoublepage
    \endgroup

    % list of tables
    \listoftables
    \thispagestyle{empty}

    % preface page (OPTIONAL)
    \ifprefacepage
	    \@prefacepage
    \fi

    % Prepare things for the rest of the document
    \clearpage
    \thispagestyle{empty}
}

%
% Configure the mainmatter and backmatter pages
%
\renewcommand{\mainmatter}{
	% just set the style of the pages to be fancy
    \pagestyle{fancy}
}
\renewcommand{\backmatter}{
	% just set the style of the pages to be fancy
    \pagestyle{fancy}

    % and also, make the empty page style permanent, since
    % when a \chapter command is invoked, the command
    % \thispagestyle{plain} is invoked
    % (thanks to http://tex.stackexchange.com/a/19741)
    %\patchcmd{\chapter}{plain}{empty}{}{}
}

%
% Configure chapter printing and alignment (centered)
%
\def\@makechapterhead#1{%
  %\vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \interlinepenalty\@M
    \ifappendixstarted
       \large \centering \bfseries
       \ifenglishlanguage
           APPENDIX
       \else
           ΠΑΡΑΡΤΗΜΑ
       \fi
       \thechapter. \MakeTextUppercase{#1}
    \else
	   \large \centering \bfseries \thechapter. \MakeTextUppercase{#1}
    \fi
    \par\nobreak
    \vskip 20\p@
  }}

\chaptertitlefont{\vspace*{-2.38cm} \large \centering \MakeTextUppercase}
\sectionfont{\normalsize}
\subsectionfont{\normalsize}
\subsubsectionfont{\normalsize}
\paragraphfont{\normalsize}
\subparagraphfont{\normalsize}

% use capital letters for chapters in the ToC as well
% TODO

%
% Rename Bibliography to References
%
%\renewcommand{\bibname}{REFERENCES}
\setlocalecaption{english}{bib}{REFERENCES}
\setlocalecaption{greek}{bib}{ΑΝΑΦΟΡΕΣ}

%
% Set ToC depth
%
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

%
% No indentation for paragraphs
%
\setlength{\parindent}{0pt}

%
% Paragraph spacing should be
%
\setlength{\parskip}{6pt}

%
% Line spacing should be one line
%
\linespread{1}

%
% Configure header and footer
%
%\pagestyle{fancy}
\fancyhf{}
\ifoneside
  \fancyhead[L]{{\scriptsize
          \ifenglishlanguage
              \@titleEn{}
          \else
              \@titleGr{}
          \fi}}
  \fancyfoot[R]{{\scriptsize \thepage}}
  \fancyfoot[L]{{\scriptsize
          \ifenglishlanguage
              \@authorFirstAbrEn{} \@authorLastEn{}
          \else
              \@authorFirstAbrGr{} \@authorLastGr{}
          \fi}}
\else
  \fancyhead[LE,RO]{{\scriptsize
          \ifenglishlanguage
              \@titleEn{}
          \else
              \@titleGr{}
          \fi}}
  \fancyfoot[C]{{\scriptsize \thepage}}
  \fancyfoot[LE,RO]{{\scriptsize
          \ifenglishlanguage
              \@authorFirstAbrEn{} \@authorLastEn{}
          \else
              \@authorFirstAbrGr{} \@authorLastGr{}
          \fi}}
\fi

% needed redefinition, because commands like \chapter call
% \thispagestyle{plain} automatically
\fancypagestyle{plain} {
  \fancyhf{}
  \ifoneside
  \fancyhead[L]{{\scriptsize
          \ifenglishlanguage
          \@titleEn{}
          \else
          \@titleGr{}
          \fi}}
  \fancyfoot[R]{{\scriptsize \thepage}}
  \fancyfoot[L]{{\scriptsize
          \ifenglishlanguage
          \@authorFirstAbrEn{} \@authorLastEn{}
          \else
          \@authorFirstAbrGr{} \@authorLastGr{}
          \fi}}
  \else
  \fancyhead[LE,RO]{{\scriptsize
          \ifenglishlanguage
          \@titleEn{}
          \else
          \@titleGr{}
          \fi}}
  \fancyfoot[C]{{\scriptsize \thepage}}
  \fancyfoot[LE,RO]{{\scriptsize
          \ifenglishlanguage
          \@authorFirstAbrEn{} \@authorLastEn{}
          \else
          \@authorFirstAbrGr{} \@authorLastGr{}
          \fi}}
  \fi
}
\renewcommand{\headrulewidth}{0pt}

%
% Configure the first page of ToC to have an empty pagestyle
% (thanks to http://tex.stackexchange.com/a/5789)
%
\AtBeginDocument{\addtocontents{toc}{\protect\thispagestyle{empty}}}
