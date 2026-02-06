;;; semantic-niche.el --- Pure Elisp Evolutionary Niche Simulator

(require 'cl-lib)

(defgroup semantic-niche nil
  "Configuration for the Semantic Niche Simulator"
  :group 'games)

;; --- Configuration Parameters ---
(defcustom sn-pop-size 50 "Population size" :type 'integer)
(defcustom sn-mutation-rate 0.12 "Mutation probability (0.0 - 1.0)" :type 'float)
(defvar sn-alphabet "abcdefghijklmnopqrstuvwxyz ")
(defvar sn-canvas-width 70)
(defvar sn-canvas-height 30)

;; --- State Variables ---
(defvar sn--current-pop nil)
(defvar sn--target "" )
(defvar sn--timer nil)
(defvar sn--generation 0)

;; --- Core Logic: Evolutionary Algorithm ---

(defun sn--random-string (len)
  "Generate a random initial string."
  (let ((res ""))
    (dotimes (_ len res)
      (setq res (concat res (string (elt sn-alphabet (random (length sn-alphabet)))))))))

(defun sn--get-fitness (ind target)
  "Calculate fitness: character matching ratio (0.0 to 1.0)."
  (let ((score 0)
        (len (length target)))
    (cl-loop for i from 0 below len
             do (when (= (elt ind i) (elt target i))
                  (cl-incf score)))
    (/ (float score) len)))

(defun sn--mutate (ind)
  "Randomly mutate characters in the individual string."
  (let ((new-ind (copy-sequence ind)))
    (cl-loop for i from 0 below (length new-ind)
             do (when (< (/ (random 1000) 1000.0) sn-mutation-rate)
                  (aset new-ind i (elt sn-alphabet (random (length sn-alphabet))))))
    new-ind))

(defun sn--evolve-step ()
  "Evolve the population by selection and mutation."
  (let* ((fitness-pairs (mapcar (lambda (ind) (cons (sn--get-fitness ind sn--target) ind)) sn--current-pop))
         ;; Sort by fitness descending
         (sorted-pairs (sort fitness-pairs (lambda (a b) (> (car a) (car b)))))
         (best (cdr (car sorted-pairs)))
         (new-pop (list best))) ; Elitism: preserve the best individual
    
    ;; Fill the next generation
    (while (< (length new-pop) sn-pop-size)
      (let* ((parent-idx (random (/ sn-pop-size 2))) ; Tournament selection from top 50%
             (parent (cdr (nth parent-idx sorted-pairs))))
        (push (sn--mutate parent) new-pop)))
    (setq sn--current-pop new-pop)))

;; --- Rendering Logic: Polar ASCII Visualization ---

(defun sn--render ()
  "Render the evolution field using polar mapping in a buffer."
  (with-current-buffer (get-buffer-create "*Semantic Niche*")
    (let ((inhibit-read-only t)
          (center-x (/ sn-canvas-width 2))
          (center-y (/ sn-canvas-height 2)))
      (erase-buffer)
      
      ;; Draw Header
      (insert (propertize (format " GENERATION: %04d | TARGET: [%s]\n" sn--generation sn--target) 
                          'face '(:weight bold :foreground "#00FFCC")))
      (insert (make-string sn-canvas-width ?-) "\n")
      
      ;; Initialize blank canvas
      (let ((canvas (make-vector sn-canvas-height nil)))
        (dotimes (i sn-canvas-height)
          (aset canvas i (make-string sn-canvas-width ?\s)))
        
        ;; Map population to canvas
        (cl-loop for i from 0 below (length sn--current-pop)
                 for ind = (nth i sn--current-pop)
                 for fit = (sn--get-fitness ind sn--target)
                 do (let* (;; Radius R: Higher fitness means smaller radius (convergence to center)
                           (radius (* (- 1.1 fit) (/ sn-canvas-height 2.2)))
                           (angle (* (/ (float i) sn-pop-size) 2 pi))
                           ;; Polar to Cartesian conversion (x2 multiplier for character aspect ratio)
                           (x (truncate (+ center-x (* radius (cos angle) 2.2)))) 
                           (y (truncate (+ center-y (* radius (sin angle))))))
                      (when (and (>= y 0) (< y sn-canvas-height)
                                 (>= x 0) (< x sn-canvas-width))
                        (let ((row (aref canvas y))
                              ;; Display a random char from the individual or '*' if fit is high
                              (char (if (> fit 0.8) (elt ind (random (length ind))) ?.)))
                          (aset row x char)))))
        
        ;; Output canvas rows
        (dotimes (i sn-canvas-height)
          (insert (aref canvas i) "\n")))
      
      (insert "\n" (make-string sn-canvas-width ?-) "\n")
      (insert (propertize " BEST ADAPTATION: " 'face '(:italic t))
              (propertize (car sn--current-pop) 'face '(:foreground "yellow" :weight bold)))
      (insert (format "\n PROGRESS: [%s%s] %.1f%%" 
                      (make-string (truncate (* (sn--get-fitness (car sn--current-pop) sn--target) 20)) ?#)
                      (make-string (- 20 (truncate (* (sn--get-fitness (car sn--current-pop) sn--target) 20))) ?-)
                      (* (sn--get-fitness (car sn--current-pop) sn--target) 100))))))

;; --- Controller ---

(defun sn--loop ()
  "Main animation loop."
  (setq sn--generation (1+ sn--generation))
  (sn--evolve-step)
  (sn--render)
  ;; Check for convergence
  (when (string= (car sn--current-pop) sn--target)
    (cancel-timer sn--timer)
    (message "Evolution Successful! Converged at Gen %d" sn--generation)))

;;;###autoload
(defun semantic-niche-start (target)
  "Start the semantic space evolution simulation."
  (interactive "sEnter Target Word: ")
  (when (or (null target) (string= target "")) (setq target "emacs"))
  
  ;; Initialize State
  (setq sn--target (downcase target))
  (setq sn--generation 0)
  (setq sn--current-pop (cl-loop repeat sn-pop-size collect (sn--random-string (length target))))
  
  ;; Setup Buffer
  (switch-to-buffer (get-buffer-create "*Semantic Niche*"))
  (setq cursor-type nil)
  (read-only-mode 1)
  
  ;; Start Async Timer (20 updates per second)
  (when sn--timer (cancel-timer sn--timer))
  (setq sn--timer (run-with-timer 0 0.05 #'sn--loop)))

(defun semantic-niche-stop ()
  "Stop the simulation."
  (interactive)
  (when sn--timer
    (cancel-timer sn--timer)
    (message "Evolution stopped.")))

(provide 'semantic-niche)
