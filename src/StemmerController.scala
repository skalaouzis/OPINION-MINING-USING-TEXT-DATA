/*******************************************************************************
  * Authors: Giorgos Zafaras - Stavros Kalaouzis                               *
  * Year: 2016-2017                                                            *
  * Aristotle University of Thessaloniki: Computer Science Department          *
  ******************************************************************************/


object StemmerController {

  def stem(text: String): String =
    {
      var stemmedOpinion = ""
      var stemmer = new Stemmer()

      for (k <- text.trim().split(" ")) {
        stemmer.add(k)
        if (stemmer.b.length > 2) {
          stemmer.step1()
          stemmer.step2()
          stemmer.step3()
          stemmer.step4()
          stemmer.step5a()
          stemmer.step5b()

          stemmedOpinion += " " + stemmer.b
        }
      }
      return stemmedOpinion.trim()
    }
}